//! Shape Manipulation Operations
//!
//! This module contains tensor operations that manipulate the shape and structure of tensors,
//! including reshaping, expanding/squeezing dimensions, broadcasting, and flattening operations.
//!
//! These operations primarily work with tensor metadata and layout rather than the actual data values,
//! making them efficient for reorganizing tensor structures without expensive data copying.

#[cfg(feature = "gpu")]
use crate::gpu::buffer::GpuBuffer;
use crate::tensor::TensorStorage;
use crate::{Result, Tensor, TensorError};
use num_traits::Zero;
use scirs2_autograd::ndarray::IxDyn;
#[cfg(feature = "gpu")]
use wgpu::util::DeviceExt;

/// Reshape a tensor to a new shape
///
/// Changes the dimensions of a tensor while preserving the total number of elements.
/// The new shape must have the same total size as the original tensor.
///
/// # Arguments
/// * `tensor` - Input tensor to reshape
/// * `shape` - New shape as slice of dimensions
///
/// # Returns
/// A new tensor with the specified shape containing the same elements
///
/// # Errors
/// Returns error if the total size of the new shape doesn't match the original tensor size
pub fn reshape<T>(tensor: &Tensor<T>, shape: &[usize]) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + Send + Sync + 'static,
{
    let total_size: usize = shape.iter().product();
    let tensor_size = tensor.shape().size();

    if total_size != tensor_size {
        return Err(TensorError::invalid_argument(format!(
            "Cannot reshape tensor of size {tensor_size} to shape {shape:?} (size {total_size})"
        )));
    }

    match &tensor.storage {
        TensorStorage::Cpu(array) => {
            // For CPU tensors, we can use ndarray's reshape
            let new_array = array
                .clone()
                .into_shape_with_order(IxDyn(shape))
                .map_err(|e| TensorError::invalid_argument(format!("Reshape failed: {e}")))?;

            Ok(Tensor::from_array(new_array))
        }
        #[cfg(feature = "gpu")]
        TensorStorage::Gpu(gpu_buffer) => gpu_reshape_dispatch(gpu_buffer, shape),
    }
}

/// Add a dimension of size 1 at the specified axis
///
/// Inserts a new dimension of size 1 at the given axis position. This operation
/// doesn't change the data but adds a singleton dimension to the tensor shape.
///
/// # Arguments
/// * `tensor` - Input tensor to expand
/// * `axis` - Position where to insert the new dimension (0-indexed)
///
/// # Returns
/// A new tensor with an additional dimension of size 1 at the specified axis
///
/// # Errors
/// Returns error if axis is out of range for the tensor
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

/// Remove dimensions of size 1
///
/// Removes dimensions of size 1 from the tensor. Can either remove all singleton dimensions
/// or only specific axes if provided.
///
/// # Arguments
/// * `tensor` - Input tensor to squeeze
/// * `axes` - Optional specific axes to squeeze. If None, removes all dimensions of size 1
///
/// # Returns
/// A new tensor with singleton dimensions removed
///
/// # Errors
/// Returns error if specified axes are out of range or don't have size 1
pub fn squeeze<T>(tensor: &Tensor<T>, axes: Option<&[usize]>) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + Send + Sync + 'static,
{
    let shape = tensor.shape().dims();
    let mut new_shape = Vec::new();
    let mut removed_axes = Vec::new();

    if let Some(axes) = axes {
        // Remove specific axes
        for &axis in axes {
            if axis >= shape.len() {
                return Err(TensorError::invalid_argument(format!(
                    "Axis {axis} out of range for tensor of rank {}",
                    shape.len()
                )));
            }
            if shape[axis] != 1 {
                return Err(TensorError::invalid_argument(format!(
                    "Cannot squeeze axis {axis} of size {}",
                    shape[axis]
                )));
            }
        }

        for (i, &dim) in shape.iter().enumerate() {
            if !axes.contains(&i) {
                new_shape.push(dim);
            } else {
                removed_axes.push(i);
            }
        }
    } else {
        // Remove all axes of size 1
        for (i, &dim) in shape.iter().enumerate() {
            if dim != 1 {
                new_shape.push(dim);
            } else {
                removed_axes.push(i);
            }
        }
    }

    reshape(tensor, &new_shape)
}

/// Broadcast tensor to a target shape
///
/// Expands the tensor to match a target shape following NumPy broadcasting rules.
/// Dimensions are aligned from the right, and singleton dimensions can be expanded.
///
/// # Arguments
/// * `tensor` - Input tensor to broadcast
/// * `target_shape` - Target shape to broadcast to
///
/// # Returns
/// A new tensor broadcasted to the target shape
///
/// # Errors
/// Returns error if broadcasting is not possible according to NumPy rules
pub fn broadcast_to<T>(tensor: &Tensor<T>, target_shape: &[usize]) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    let current_shape = tensor.shape().dims();

    // Check if broadcasting is possible
    if current_shape.len() > target_shape.len() {
        return Err(TensorError::invalid_shape_simple(format!(
            "Cannot broadcast tensor with {} dimensions to {} dimensions ",
            current_shape.len(),
            target_shape.len()
        )));
    }

    // Check dimension compatibility
    let offset = target_shape.len() - current_shape.len();
    for (i, &current_dim) in current_shape.iter().enumerate() {
        let target_dim = target_shape[i + offset];
        if current_dim != 1 && current_dim != target_dim {
            return Err(TensorError::shape_mismatch(
                "broadcast_to",
                &format!("compatible dimensions for broadcasting to {target_shape:?}"),
                &format!("current shape {current_shape:?}"),
            ));
        }
    }

    match &tensor.storage {
        TensorStorage::Cpu(array) => {
            // Use ndarray's broadcasting capability
            let target_dim = IxDyn(target_shape);
            let broadcasted = array.broadcast(target_dim).ok_or_else(|| {
                TensorError::invalid_shape_simple(format!(
                    "Broadcasting to shape {target_shape:?} failed "
                ))
            })?;

            // Convert to owned array
            let result = broadcasted.to_owned();
            Ok(Tensor::from_array(result))
        }
        #[cfg(feature = "gpu")]
        TensorStorage::Gpu(_) => gpu_broadcast_to_dispatch(tensor, target_shape),
    }
}

/// Expand tensor dimensions by adding new axes
///
/// Broadcasts the input tensor to match the shape of the target tensor.
/// This is a convenience function equivalent to `broadcast_to(tensor, target.shape())`.
///
/// # Arguments
/// * `tensor` - Input tensor to expand
/// * `target` - Target tensor whose shape to match
///
/// # Returns
/// A new tensor with the same shape as the target tensor
pub fn expand_as<T>(tensor: &Tensor<T>, target: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    broadcast_to(tensor, target.shape().dims())
}

/// Unsqueeze operation - add dimensions of size 1
///
/// Adds new dimensions of size 1 at the specified axes. Unlike expand_dims which adds
/// a single dimension, unsqueeze can add multiple dimensions at once.
///
/// # Arguments
/// * `tensor` - Input tensor to unsqueeze
/// * `axes` - Axes positions where to insert dimensions of size 1
///
/// # Returns
/// A new tensor with additional singleton dimensions at the specified axes
///
/// # Errors
/// Returns error if axes are out of range or contain duplicates
pub fn unsqueeze<T>(tensor: &Tensor<T>, axes: &[usize]) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + Send + Sync + 'static,
{
    let original_shape = tensor.shape().dims();
    let new_rank = original_shape.len() + axes.len();

    // Validate axes
    for &axis in axes {
        if axis > new_rank {
            return Err(TensorError::invalid_argument(format!(
                "Axis {axis} out of range for new tensor of rank {new_rank}"
            )));
        }
    }

    // Check for duplicate axes
    let mut sorted_axes = axes.to_vec();
    sorted_axes.sort();
    for i in 1..sorted_axes.len() {
        if sorted_axes[i] == sorted_axes[i - 1] {
            return Err(TensorError::invalid_argument(format!(
                "Duplicate axis {} in unsqueeze operation ",
                sorted_axes[i]
            )));
        }
    }

    // Build new shape by inserting dimensions of size 1
    let mut new_shape = Vec::with_capacity(new_rank);
    let mut original_idx = 0;

    for i in 0..new_rank {
        if sorted_axes.contains(&i) {
            new_shape.push(1);
        } else if original_idx < original_shape.len() {
            new_shape.push(original_shape[original_idx]);
            original_idx += 1;
        }
    }

    // Reshape the tensor
    reshape(tensor, &new_shape)
}

/// Flatten a tensor into a 1D tensor
///
/// This operation reshapes a tensor of any shape into a 1-dimensional tensor
/// containing the same elements in row-major (C-style) order.
///
/// # Arguments
/// * `tensor` - Input tensor to flatten
///
/// # Returns
/// A 1D tensor containing all elements from the input tensor
///
/// # Example
/// ```rust
/// use tenflowers_core::{Tensor, ops::flatten};
/// let tensor = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
/// let flattened = flatten(&tensor).unwrap();
/// assert_eq!(flattened.shape().dims(), &[4]);
/// ```
pub fn flatten<T>(tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + Send + Sync + 'static,
{
    let total_size = tensor.shape().size();
    reshape(tensor, &[total_size])
}

/// GPU reshape dispatch for supported types
#[cfg(feature = "gpu")]
fn gpu_reshape_dispatch<T>(
    gpu_buffer: &crate::gpu::buffer::GpuBuffer<T>,
    shape: &[usize],
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

        let result_buffer = crate::gpu::ops::execute_reshape(gpu_buffer_f32, shape)?;

        // Cast result back to T
        let result_buffer_t = unsafe {
            std::mem::transmute::<
                crate::gpu::buffer::GpuBuffer<f32>,
                crate::gpu::buffer::GpuBuffer<T>,
            >(result_buffer)
        };

        Ok(Tensor::from_gpu_buffer(
            result_buffer_t,
            crate::Shape::from_slice(shape),
        ))
    } else {
        Err(TensorError::unsupported_operation_simple(format!(
            "GPU reshape only supports f32, got {}",
            std::any::type_name::<T>()
        )))
    }
}

/// GPU broadcast_to dispatch for supported types
#[cfg(feature = "gpu")]
fn gpu_broadcast_to_dispatch<T>(tensor: &Tensor<T>, target_shape: &[usize]) -> Result<Tensor<T>>
where
    T: bytemuck::Pod + bytemuck::Zeroable + Clone + Default + Send + Sync + 'static,
{
    let input_shape = tensor.shape().dims();
    let total_size = target_shape.iter().product::<usize>();

    if let TensorStorage::Gpu(gpu_buffer) = &tensor.storage {
        let gpu_ctx = crate::device::context::get_gpu_context(gpu_buffer.device_enum().id())?;

        // Create output buffer
        let result_buffer = gpu_ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("broadcast_output"),
            size: (total_size * std::mem::size_of::<T>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create shape buffers
        let input_shape_data: Vec<u32> = input_shape.iter().map(|&s| s as u32).collect();
        let input_shape_buffer =
            gpu_ctx
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("broadcast_input_shape"),
                    contents: bytemuck::cast_slice(&input_shape_data),
                    usage: wgpu::BufferUsages::STORAGE,
                });

        let output_shape_data: Vec<u32> = target_shape.iter().map(|&s| s as u32).collect();
        let output_shape_buffer =
            gpu_ctx
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("broadcast_output_shape"),
                    contents: bytemuck::cast_slice(&output_shape_data),
                    usage: wgpu::BufferUsages::STORAGE,
                });

        // Create uniform buffer
        let info = [
            input_shape.len() as u32,
            target_shape.len() as u32,
            total_size as u32,
            0u32,
        ];
        let uniform_buffer = gpu_ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("broadcast_uniform"),
                contents: bytemuck::cast_slice(&info),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Create compute shader
        let shader = gpu_ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("broadcast_shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../../gpu/shaders/manipulation_ops2.wgsl").into(),
                ),
            });

        // Create bind group layout
        let bind_group_layout =
            gpu_ctx
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("broadcast_bind_group_layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        // Create bind group
        let bind_group = gpu_ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("broadcast_bind_group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: gpu_buffer.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: result_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: input_shape_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: output_shape_buffer.as_entire_binding(),
                    },
                ],
            });

        // Create compute pipeline
        let pipeline_layout =
            gpu_ctx
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("broadcast_pipeline_layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let compute_pipeline =
            gpu_ctx
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("broadcast_pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some("broadcast_op"),
                    cache: None,
                    compilation_options: Default::default(),
                });

        // Execute compute shader
        let mut encoder = gpu_ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("broadcast_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("broadcast_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&compute_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups((total_size as u32 + 63) / 64, 1, 1);
        }

        gpu_ctx.queue.submit(std::iter::once(encoder.finish()));

        let result_buffer = GpuBuffer::from_raw_buffer(
            result_buffer,
            gpu_ctx.device.clone(),
            gpu_ctx.queue.clone(),
            gpu_buffer.device_enum(),
            total_size,
        );

        Ok(Tensor::from_gpu_buffer(
            result_buffer,
            crate::Shape::from_slice(target_shape),
        ))
    } else {
        Err(TensorError::device_mismatch("reshape", "GPU", "CPU"))
    }
}
