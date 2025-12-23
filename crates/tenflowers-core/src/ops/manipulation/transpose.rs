//! Transpose and permutation operations for tensors
//!
//! This module contains functions for transposing tensors, flipping axes,
//! rolling elements, and other permutation-based tensor manipulations.
//!
//! All operations support both CPU and GPU execution paths when the GPU
//! feature is enabled.

#[cfg(feature = "gpu")]
use crate::gpu::buffer::GpuBuffer;
use crate::tensor::TensorStorage;
use crate::{Result, Tensor, TensorError};
use scirs2_core::ndarray::{ArrayD, IxDyn};
use scirs2_core::numeric::Zero;
#[cfg(feature = "gpu")]
use wgpu::util::DeviceExt;

/// Transpose a tensor (reverse all dimensions)
pub fn transpose<T>(tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + Send + Sync + 'static,
{
    transpose_axes(tensor, None)
}

/// Transpose a tensor with specified axis permutation
pub fn transpose_axes<T>(tensor: &Tensor<T>, axes: Option<&[usize]>) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + Send + Sync + 'static,
{
    let rank = tensor.shape().rank();

    // Default axes: reverse all dimensions
    let default_axes: Vec<usize> = (0..rank).rev().collect();
    let axes = axes.unwrap_or(&default_axes);

    // Validate axes
    if axes.len() != rank {
        return Err(TensorError::invalid_argument(format!(
            "Axes length {} does not match tensor rank {rank}",
            axes.len()
        )));
    }

    // Check for valid permutation
    let mut seen = vec![false; rank];
    for &axis in axes {
        if axis >= rank {
            return Err(TensorError::invalid_argument(format!(
                "Axis {axis} out of range for tensor of rank {rank}"
            )));
        }
        if seen[axis] {
            return Err(TensorError::invalid_argument(format!(
                "Duplicate axis: {axis}"
            )));
        }
        seen[axis] = true;
    }

    match &tensor.storage {
        TensorStorage::Cpu(array) => {
            // Use ndarray's permuted_axes
            let new_array = array.clone().permuted_axes(axes);
            Ok(Tensor::from_array(new_array))
        }
        #[cfg(feature = "gpu")]
        TensorStorage::Gpu(gpu_buffer) => {
            gpu_transpose_dispatch(gpu_buffer, tensor.shape().dims(), axes)
        }
    }
}

/// Roll operation - roll tensor elements along specified axes
pub fn roll<T>(tensor: &Tensor<T>, shift: isize, axis: Option<usize>) -> Result<Tensor<T>>
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
                // Roll along a specific axis
                let axis_size = tensor.shape().dims()[axis] as isize;
                let normalized_shift = ((shift % axis_size) + axis_size) % axis_size;

                if normalized_shift == 0 {
                    return Ok(tensor.clone());
                }

                let mut result = array.clone();
                let shape = tensor.shape().dims();

                // Create indices for iteration
                let mut indices = vec![0; shape.len()];

                loop {
                    // Calculate source indices (where to copy from)
                    let mut src_indices = indices.clone();
                    src_indices[axis] = ((indices[axis] as isize - normalized_shift + axis_size)
                        % axis_size) as usize;

                    // Copy the value
                    result[IxDyn(&indices)] = array[IxDyn(&src_indices)].clone();

                    // Increment indices
                    let mut carry = true;
                    for i in (0..shape.len()).rev() {
                        if carry {
                            indices[i] += 1;
                            if indices[i] < shape[i] {
                                carry = false;
                            } else {
                                indices[i] = 0;
                            }
                        }
                    }
                    if carry {
                        break;
                    }
                }

                Ok(Tensor::from_array(result))
            } else {
                // Roll the flattened tensor
                let flat = array
                    .view()
                    .into_shape_with_order((array.len(),))
                    .map_err(|e| {
                        TensorError::invalid_argument(format!("Failed to flatten: {e}"))
                    })?;

                let len = flat.len() as isize;
                let normalized_shift = ((shift % len) + len) % len;

                if normalized_shift == 0 {
                    return Ok(tensor.clone());
                }

                let mut rolled = Vec::with_capacity(flat.len());

                // Copy elements in rolled order
                for i in 0..flat.len() {
                    let src_idx = ((i as isize - normalized_shift + len) % len) as usize;
                    rolled.push(flat[src_idx].clone());
                }

                let rolled_array =
                    ArrayD::from_shape_vec(array.raw_dim(), rolled).map_err(|e| {
                        TensorError::invalid_argument(format!("Failed to create rolled array: {e}"))
                    })?;

                Ok(Tensor::from_array(rolled_array))
            }
        }
        #[cfg(feature = "gpu")]
        TensorStorage::Gpu(_) => gpu_roll_dispatch(tensor, shift, axis),
    }
}

/// Flip (reverse) tensor along specified axes
pub fn flip<T>(tensor: &Tensor<T>, axes: &[usize]) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    let shape = tensor.shape();
    let rank = shape.rank();

    // Validate axes
    for &axis in axes {
        if axis >= rank {
            return Err(TensorError::invalid_argument(format!(
                "Axis {axis} out of range for tensor of rank {rank}"
            )));
        }
    }

    match &tensor.storage {
        TensorStorage::Cpu(array) => {
            let mut result = array.clone();

            // Flip along each specified axis
            for &axis in axes {
                result.invert_axis(scirs2_core::ndarray::Axis(axis));
            }

            Ok(Tensor::from_array(result))
        }
        #[cfg(feature = "gpu")]
        TensorStorage::Gpu(_) => {
            // For GPU, we need to flip along each axis sequentially
            let mut result = tensor.clone();
            for &axis in axes {
                result = gpu_flip_dispatch(&result, axis)?;
            }
            Ok(result)
        }
    }
}

/// GPU transpose dispatch for supported types
#[cfg(feature = "gpu")]
fn gpu_transpose_dispatch<T>(
    gpu_buffer: &crate::gpu::buffer::GpuBuffer<T>,
    input_shape: &[usize],
    axes: &[usize],
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

        // Calculate output shape
        let output_shape: Vec<usize> = axes.iter().map(|&i| input_shape[i]).collect();
        let output_len: usize = output_shape.iter().product();

        let result_buffer =
            crate::gpu::ops::execute_transpose(gpu_buffer_f32, axes, input_shape, output_len)?;

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
            "GPU transpose only supports f32, got {}",
            std::any::type_name::<T>()
        )))
    }
}

/// GPU roll dispatch for supported types
#[cfg(feature = "gpu")]
fn gpu_roll_dispatch<T>(tensor: &Tensor<T>, shift: isize, axis: Option<usize>) -> Result<Tensor<T>>
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

        // Convert single values to arrays for GPU function
        let shifts = &[shift as i32];
        let axes = axis.as_ref().map(|a| std::slice::from_ref(a));

        let result_buffer =
            crate::gpu::ops::execute_roll(gpu_buffer, shifts, axes, tensor.shape().dims())?;

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
            "GPU roll only supports f32, got {}",
            std::any::type_name::<T>()
        )))
    }
}

/// GPU flip dispatch for supported types
#[cfg(feature = "gpu")]
fn gpu_flip_dispatch<T>(tensor: &Tensor<T>, axis: usize) -> Result<Tensor<T>>
where
    T: bytemuck::Pod + bytemuck::Zeroable + Clone + Default + Send + Sync + 'static,
{
    let shape = tensor.shape().dims();
    let total_size = shape.iter().product::<usize>();

    if let TensorStorage::Gpu(gpu_buffer) = &tensor.storage {
        let gpu_ctx = crate::device::context::get_gpu_context(gpu_buffer.device_enum().id())?;

        // Create output buffer
        let result_buffer = gpu_ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flip_output"),
            size: (total_size * std::mem::size_of::<T>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create shape buffer
        let shape_data: Vec<u32> = shape.iter().map(|&s| s as u32).collect();
        let shape_buffer = gpu_ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("flip_shape"),
                contents: bytemuck::cast_slice(&shape_data),
                usage: wgpu::BufferUsages::STORAGE,
            });

        // Create uniform buffer
        let info = [axis as u32, shape.len() as u32, total_size as u32, 0u32];
        let uniform_buffer = gpu_ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("flip_uniform"),
                contents: bytemuck::cast_slice(&info),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Create compute shader
        let shader = gpu_ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("flip_shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../../gpu/shaders/manipulation_ops2.wgsl").into(),
                ),
            });

        // Create bind group layout
        let bind_group_layout =
            gpu_ctx
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("flip_bind_group_layout"),
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
                    ],
                });

        // Create bind group
        let bind_group = gpu_ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("flip_bind_group"),
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
                        resource: shape_buffer.as_entire_binding(),
                    },
                ],
            });

        // Create compute pipeline
        let pipeline_layout =
            gpu_ctx
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("flip_pipeline_layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let compute_pipeline =
            gpu_ctx
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("flip_pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some("flip_op"),
                    cache: None,
                    compilation_options: Default::default(),
                });

        // Execute compute shader
        let mut encoder = gpu_ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("flip_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("flip_pass"),
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
            crate::Shape::from_slice(shape),
        ))
    } else {
        Err(TensorError::device_mismatch("transpose", "GPU", "CPU"))
    }
}
