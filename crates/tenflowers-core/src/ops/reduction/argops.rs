//! Argument-based reduction operations
//!
//! This module contains reduction operations that return indices rather than values,
//! such as argmax, argmin, and topk operations. These functions are useful for
//! finding the positions of specific elements in tensors.

use super::common::normalize_axis;
use crate::tensor::TensorStorage;
use crate::{Result, Tensor, TensorError};
use scirs2_core::ndarray::{ArrayD, Axis};

/// Compute argmax along axes
pub fn argmax<T>(x: &Tensor<T>, axis: Option<i32>, keepdims: bool) -> Result<Tensor<usize>>
where
    T: Clone + Default + PartialOrd + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    match &x.storage {
        TensorStorage::Cpu(arr) => {
            if let Some(axis) = axis {
                let axis = normalize_axis(axis, x.shape().rank() as i32)?;

                // Map along the axis to find argmax indices
                let indices = arr.map_axis(Axis(axis), |view| {
                    view.iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(idx, _)| idx)
                        .unwrap_or(0)
                });

                let result = if keepdims {
                    indices.insert_axis(Axis(axis))
                } else {
                    indices
                };

                Ok(Tensor::from_array(result))
            } else {
                // Find argmax of flattened array
                let (max_idx, _) = arr
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or((0, &T::default()));

                let result = if keepdims {
                    ArrayD::from_elem(vec![1; x.shape().rank()], max_idx)
                } else {
                    ArrayD::from_elem(vec![], max_idx)
                };
                Ok(Tensor::from_array(result))
            }
        }
        #[cfg(feature = "gpu")]
        TensorStorage::Gpu(gpu_buffer) => {
            use crate::gpu::ops::{execute_axis_reduction_op, ReductionOp};

            // For argmax, we need to handle the result as indices (usize) but GPU returns f32
            // Calculate output shape and size
            let result_shape = if let Some(axis) = axis {
                let mut shape = x.shape().dims().to_vec();
                let normalized_axis = normalize_axis(axis, x.shape().rank() as i32)?;
                if keepdims {
                    shape[normalized_axis] = 1;
                } else {
                    shape.remove(normalized_axis);
                }
                shape
            } else if keepdims {
                vec![1; x.shape().rank()]
            } else {
                vec![]
            };

            let output_len = if result_shape.is_empty() {
                1
            } else {
                result_shape.iter().product()
            };

            // Convert axis to the format expected by GPU operations
            let axes_array;
            let gpu_axes = if let Some(axis) = axis {
                let normalized_axis = normalize_axis(axis, x.shape().rank() as i32)?;
                axes_array = vec![normalized_axis as i32];
                Some(axes_array.as_slice())
            } else {
                None
            };

            let result_buffer = execute_axis_reduction_op(
                gpu_buffer,
                ReductionOp::ArgMax,
                x.shape().dims(),
                gpu_axes,
                keepdims,
                output_len,
            )?;

            // The GPU returns f32 values representing indices, we need to convert them to usize
            // For now, we'll create a new tensor and handle the type conversion
            let f32_tensor =
                Tensor::from_gpu_buffer(result_buffer, crate::Shape::new(result_shape.clone()));

            // Convert f32 indices to usize indices by reading from GPU and converting
            match &f32_tensor.storage {
                TensorStorage::Gpu(gpu_buf) => {
                    // For now, fall back to CPU for the final conversion
                    // In a production system, this would be optimized to stay on GPU
                    let cpu_tensor = f32_tensor.to_device(crate::Device::Cpu)?;
                    match &cpu_tensor.storage {
                        TensorStorage::Cpu(arr) => {
                            let indices: Vec<usize> = arr
                                .iter()
                                .map(|&f| {
                                    // Safe conversion from f32 to usize for indices
                                    if let Ok(f_val) = bytemuck::try_cast::<T, f32>(f) {
                                        f_val as usize
                                    } else {
                                        0 // fallback value
                                    }
                                })
                                .collect();
                            let indices_tensor = Tensor::from_vec(indices, &result_shape)?;
                            Ok(indices_tensor)
                        }
                        _ => unreachable!(),
                    }
                }
                _ => unreachable!(),
            }
        }
    }
}

/// Compute argmin along axes
pub fn argmin<T>(x: &Tensor<T>, axis: Option<i32>, keepdims: bool) -> Result<Tensor<usize>>
where
    T: Clone + Default + PartialOrd + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    match &x.storage {
        TensorStorage::Cpu(arr) => {
            if let Some(axis) = axis {
                let axis = normalize_axis(axis, x.shape().rank() as i32)?;

                // Map along the axis to find argmin indices
                let indices = arr.map_axis(Axis(axis), |view| {
                    view.iter()
                        .enumerate()
                        .min_by(|(_, a), (_, b)| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(idx, _)| idx)
                        .unwrap_or(0)
                });

                let result = if keepdims {
                    indices.insert_axis(Axis(axis))
                } else {
                    indices
                };

                Ok(Tensor::from_array(result))
            } else {
                // Find argmin of flattened array
                let (min_idx, _) = arr
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or((0, &T::default()));

                let result = if keepdims {
                    ArrayD::from_elem(vec![1; x.shape().rank()], min_idx)
                } else {
                    ArrayD::from_elem(vec![], min_idx)
                };
                Ok(Tensor::from_array(result))
            }
        }
        #[cfg(feature = "gpu")]
        TensorStorage::Gpu(gpu_buffer) => {
            use crate::gpu::ops::{execute_axis_reduction_op, ReductionOp};

            // For argmin, we need to handle the result as indices (usize) but GPU returns f32
            // Calculate output shape and size
            let result_shape = if let Some(axis) = axis {
                let mut shape = x.shape().dims().to_vec();
                let normalized_axis = normalize_axis(axis, x.shape().rank() as i32)?;
                if keepdims {
                    shape[normalized_axis] = 1;
                } else {
                    shape.remove(normalized_axis);
                }
                shape
            } else if keepdims {
                vec![1; x.shape().rank()]
            } else {
                vec![]
            };

            let output_len = if result_shape.is_empty() {
                1
            } else {
                result_shape.iter().product()
            };

            // Convert axis to the format expected by GPU operations
            let axes_array;
            let gpu_axes = if let Some(axis) = axis {
                let normalized_axis = normalize_axis(axis, x.shape().rank() as i32)?;
                axes_array = vec![normalized_axis as i32];
                Some(axes_array.as_slice())
            } else {
                None
            };

            let result_buffer = execute_axis_reduction_op(
                gpu_buffer,
                ReductionOp::ArgMin,
                x.shape().dims(),
                gpu_axes,
                keepdims,
                output_len,
            )?;

            // The GPU returns f32 values representing indices, we need to convert them to usize
            // For now, we'll create a new tensor and handle the type conversion
            let f32_tensor =
                Tensor::from_gpu_buffer(result_buffer, crate::Shape::new(result_shape.clone()));

            // Convert f32 indices to usize indices by reading from GPU and converting
            match &f32_tensor.storage {
                TensorStorage::Gpu(gpu_buf) => {
                    // For now, fall back to CPU for the final conversion
                    // In a production system, this would be optimized to stay on GPU
                    let cpu_tensor = f32_tensor.to_device(crate::Device::Cpu)?;
                    match &cpu_tensor.storage {
                        TensorStorage::Cpu(arr) => {
                            let indices: Vec<usize> = arr
                                .iter()
                                .map(|&f| {
                                    // Safe conversion from f32 to usize for indices
                                    if let Ok(f_val) = bytemuck::try_cast::<T, f32>(f) {
                                        f_val as usize
                                    } else {
                                        0 // fallback value
                                    }
                                })
                                .collect();
                            let indices_tensor = Tensor::from_vec(indices, &result_shape)?;
                            Ok(indices_tensor)
                        }
                        _ => unreachable!(),
                    }
                }
                _ => unreachable!(),
            }
        }
    }
}

/// Find the top K largest elements along the last axis
/// Returns (values, indices) where both have shape [..., k]
pub fn topk<T>(x: &Tensor<T>, k: usize, axis: Option<i32>) -> Result<(Tensor<T>, Tensor<usize>)>
where
    T: Clone + Default + PartialOrd + Send + Sync + 'static,
{
    match &x.storage {
        TensorStorage::Cpu(arr) => {
            let shape = arr.shape();
            let default_axis = shape.len() as i32 - 1; // Default to last axis
            let axis = normalize_axis(axis.unwrap_or(default_axis), shape.len() as i32)?;
            let axis_size = shape[axis];

            if k > axis_size {
                return Err(TensorError::invalid_argument(format!(
                    "k={k} is larger than axis size={axis_size}"
                )));
            }

            // Create output shapes
            let mut values_shape = shape.to_vec();
            let mut indices_shape = shape.to_vec();
            values_shape[axis] = k;
            indices_shape[axis] = k;

            // Initialize output arrays
            let mut values_data = vec![T::default(); values_shape.iter().product()];
            let mut indices_data = vec![0usize; indices_shape.iter().product()];

            // Calculate strides for iteration
            let mut strides = vec![1; shape.len()];
            for i in (0..shape.len() - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }

            // Calculate output strides
            let mut out_strides = vec![1; values_shape.len()];
            for i in (0..values_shape.len() - 1).rev() {
                out_strides[i] = out_strides[i + 1] * values_shape[i + 1];
            }

            // Iterate over all slices along the specified axis
            let total_elements: usize = shape.iter().product();
            let slices_count = total_elements / axis_size;

            for slice_idx in 0..slices_count {
                // Calculate the starting index for this slice
                let mut coords = vec![0; shape.len()];
                let mut remaining = slice_idx;

                for (i, &dim_size) in shape.iter().enumerate() {
                    if i == axis {
                        continue;
                    }
                    let reduced_stride = total_elements / (dim_size * axis_size);
                    coords[i] = remaining / reduced_stride;
                    remaining %= reduced_stride;
                }

                // Collect values and their indices along the axis
                let mut values_with_indices: Vec<(T, usize)> = Vec::with_capacity(axis_size);
                for i in 0..axis_size {
                    coords[axis] = i;
                    let linear_idx = coords
                        .iter()
                        .zip(strides.iter())
                        .map(|(&c, &s)| c * s)
                        .sum::<usize>();

                    if let Some(val) = arr.as_slice() {
                        values_with_indices.push((val[linear_idx].clone(), i));
                    }
                }

                // Sort by value in descending order and take top k
                values_with_indices
                    .sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                values_with_indices.truncate(k);

                // Store results
                for (rank, (value, orig_idx)) in values_with_indices.iter().enumerate() {
                    coords[axis] = rank;
                    let out_linear_idx = coords
                        .iter()
                        .zip(out_strides.iter())
                        .map(|(&c, &s)| c * s)
                        .sum::<usize>();

                    values_data[out_linear_idx] = value.clone();
                    indices_data[out_linear_idx] = *orig_idx;
                }
            }

            let values_tensor = Tensor::from_vec(values_data, &values_shape)?;
            let indices_tensor = Tensor::from_vec(indices_data, &indices_shape)?;

            Ok((values_tensor, indices_tensor))
        }
        #[cfg(feature = "gpu")]
        TensorStorage::Gpu(gpu_buffer) => {
            // GPU topk implementation
            let shape = x.shape();
            let default_axis = shape.rank() as i32 - 1; // Default to last axis
            let axis = normalize_axis(axis.unwrap_or(default_axis), shape.rank() as i32)?;
            let axis_size = shape.dims()[axis];
            let total_elements: usize = shape.dims().iter().product();
            let num_slices = total_elements / axis_size;

            // Use type-safe transmute to call GPU topk with correct types
            let (values_buffer, indices_buffer) = if std::any::type_name::<T>() == "f32" {
                let gpu_buffer_f32 = unsafe {
                    std::mem::transmute::<
                        &crate::gpu::buffer::GpuBuffer<T>,
                        &crate::gpu::buffer::GpuBuffer<f32>,
                    >(gpu_buffer)
                };
                // TODO: Implement GPU topk operation
                return Err(TensorError::unsupported_operation_simple(
                    "GPU topk operation not yet implemented".to_string(),
                ));
            } else {
                return Err(TensorError::unsupported_operation_simple(format!(
                    "GPU topk only supports f32 currently, got {}",
                    std::any::type_name::<T>()
                )));
            };

            #[allow(unreachable_code)]
            // Create output tensors
            // Compute output shape: same as input but axis dimension becomes k
            let mut values_shape = shape.dims().to_vec();
            values_shape[axis] = k;
            let indices_shape = values_shape.clone();

            let values_tensor =
                Tensor::from_gpu_buffer(values_buffer, crate::Shape::new(values_shape));
            let indices_tensor =
                Tensor::from_gpu_buffer(indices_buffer, crate::Shape::new(indices_shape));

            Ok((values_tensor, indices_tensor))
        }
    }
}
