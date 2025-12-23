//! Batch Normalization Operations
//!
//! This module provides batch normalization operations for deep learning,
//! supporting both training and inference modes with optimized CPU and GPU implementations.
//! Batch normalization normalizes the input across the batch dimension for each channel,
//! which helps with training stability and convergence.

use crate::tensor::TensorStorage;
use crate::{Result, Tensor, TensorError};
use scirs2_core::ndarray::{ArrayD, IxDyn};
use scirs2_core::numeric::{Float, FromPrimitive};
#[cfg(feature = "gpu")]
use wgpu::util::DeviceExt;

/// Batch normalization for 4D tensors (NCHW format)
/// Input shape: `[batch, channels, height, width]`
/// Running mean/var shapes: `[channels]`
/// Gamma/beta shapes: `[channels]`
pub fn batch_norm<T>(
    input: &Tensor<T>,
    gamma: &Tensor<T>,
    beta: &Tensor<T>,
    running_mean: &Tensor<T>,
    running_var: &Tensor<T>,
    epsilon: T,
    training: bool,
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Float
        + FromPrimitive
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    match (
        &input.storage,
        &gamma.storage,
        &beta.storage,
        &running_mean.storage,
        &running_var.storage,
    ) {
        (
            TensorStorage::Cpu(input_arr),
            TensorStorage::Cpu(gamma_arr),
            TensorStorage::Cpu(beta_arr),
            TensorStorage::Cpu(mean_arr),
            TensorStorage::Cpu(var_arr),
        ) => {
            let input_shape = input_arr.shape();
            if input_arr.ndim() != 4 {
                return Err(TensorError::InvalidShape {
                    operation: "batch_norm".to_string(),
                    reason: "BatchNorm expects 4D input (NCHW format)".to_string(),
                    shape: Some(input_arr.shape().to_vec()),
                    context: None,
                });
            }

            let batch_size = input_shape[0];
            let channels = input_shape[1];
            let height = input_shape[2];
            let width = input_shape[3];

            // Validate parameter shapes
            if gamma_arr.shape() != [channels]
                || beta_arr.shape() != [channels]
                || mean_arr.shape() != [channels]
                || var_arr.shape() != [channels]
            {
                return Err(TensorError::ShapeMismatch {
                    operation: "batch_norm".to_string(),
                    expected: format!("parameters shape [{channels}]"),
                    got: format!(
                        "gamma: {:?}, beta: {:?}, mean: {:?}, var: {:?}",
                        gamma_arr.shape(),
                        beta_arr.shape(),
                        mean_arr.shape(),
                        var_arr.shape()
                    ),
                    context: None,
                });
            }

            let mut output = ArrayD::<T>::zeros(IxDyn(&[batch_size, channels, height, width]));

            if training {
                // Compute batch statistics
                for c in 0..channels {
                    // Calculate mean for channel c
                    let mut sum = T::zero();
                    let mut count = 0;

                    for b in 0..batch_size {
                        for h in 0..height {
                            for w in 0..width {
                                sum = sum + input_arr[[b, c, h, w]];
                                count += 1;
                            }
                        }
                    }

                    let mean = sum / T::from(count).unwrap();

                    // Calculate variance for channel c
                    let mut var_sum = T::zero();
                    for b in 0..batch_size {
                        for h in 0..height {
                            for w in 0..width {
                                let diff = input_arr[[b, c, h, w]] - mean;
                                var_sum = var_sum + (diff * diff);
                            }
                        }
                    }

                    let variance = var_sum / T::from(count).unwrap();
                    let std_dev = (variance + epsilon).sqrt();

                    // Normalize and scale
                    let gamma_val = gamma_arr[[c]];
                    let beta_val = beta_arr[[c]];

                    for b in 0..batch_size {
                        for h in 0..height {
                            for w in 0..width {
                                let normalized = (input_arr[[b, c, h, w]] - mean) / std_dev;
                                output[[b, c, h, w]] = gamma_val * normalized + beta_val;
                            }
                        }
                    }
                }
            } else {
                // Use running statistics for inference
                for c in 0..channels {
                    let mean = mean_arr[[c]];
                    let variance = var_arr[[c]];
                    let std_dev = (variance + epsilon).sqrt();
                    let gamma_val = gamma_arr[[c]];
                    let beta_val = beta_arr[[c]];

                    for b in 0..batch_size {
                        for h in 0..height {
                            for w in 0..width {
                                let normalized = (input_arr[[b, c, h, w]] - mean) / std_dev;
                                output[[b, c, h, w]] = gamma_val * normalized + beta_val;
                            }
                        }
                    }
                }
            }

            Ok(Tensor::from_array(output))
        }
        #[cfg(feature = "gpu")]
        (
            TensorStorage::Gpu(_),
            TensorStorage::Gpu(_),
            TensorStorage::Gpu(_),
            TensorStorage::Gpu(_),
            TensorStorage::Gpu(_),
        ) => batch_norm_gpu(
            input,
            gamma,
            beta,
            running_mean,
            running_var,
            epsilon,
            training,
        ),
        #[cfg(feature = "gpu")]
        _ => Err(TensorError::unsupported_operation_simple(
            "Mixed CPU/GPU batch norm not supported".to_string(),
        )),
    }
}

#[cfg(feature = "gpu")]
fn batch_norm_gpu<T>(
    input: &Tensor<T>,
    gamma: &Tensor<T>,
    beta: &Tensor<T>,
    running_mean: &Tensor<T>,
    running_var: &Tensor<T>,
    epsilon: T,
    training: bool,
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Float
        + FromPrimitive
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    use crate::gpu::buffer::GpuBuffer;
    use crate::tensor::TensorStorage;
    use std::sync::Arc;

    if let (
        TensorStorage::Gpu(input_gpu),
        TensorStorage::Gpu(gamma_gpu),
        TensorStorage::Gpu(beta_gpu),
        TensorStorage::Gpu(mean_gpu),
        TensorStorage::Gpu(var_gpu),
    ) = (
        &input.storage,
        &gamma.storage,
        &beta.storage,
        &running_mean.storage,
        &running_var.storage,
    ) {
        let device = &input_gpu.device;
        let queue = &input_gpu.queue;
        let input_shape = input.shape();

        if training {
            // Training mode: compute batch statistics on GPU
            return batch_norm_gpu_training(
                input_gpu,
                gamma_gpu,
                beta_gpu,
                mean_gpu,
                var_gpu,
                device,
                queue,
                input_shape,
                epsilon,
            );
        }

        let device = &input_gpu.device;
        let queue = &input_gpu.queue;

        let input_shape = input.shape();
        let batch_size = input_shape.dims()[0];
        let channels = input_shape.dims()[1];
        let height = input_shape.dims()[2];
        let width = input_shape.dims()[3];

        // Create output buffer
        let output_size = batch_size * channels * height * width;
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("batch_norm_output"),
            size: (output_size * std::mem::size_of::<T>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create parameters uniform buffer
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct BatchNormParams {
            batch_size: u32,
            channels: u32,
            height: u32,
            width: u32,
            epsilon: f32,
        }

        let params = BatchNormParams {
            batch_size: batch_size as u32,
            channels: channels as u32,
            height: height as u32,
            width: width as u32,
            epsilon: epsilon.to_f32().unwrap_or(1e-5),
        };

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("batch_norm_params"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Load compute shader
        let shader_source = include_str!("../../gpu/shaders/conv_ops.wgsl");
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("batch_norm_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("batch_norm_bind_group_layout"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
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
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("batch_norm_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_gpu.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: gamma_gpu.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: beta_gpu.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: mean_gpu.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: var_gpu.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Create compute pipeline
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("batch_norm_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("batch_norm_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("batch_norm_inference"),
            cache: None,
            compilation_options: Default::default(),
        });

        // Dispatch compute shader
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("batch_norm_encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("batch_norm_pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&compute_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            let workgroup_size = 64;
            let num_workgroups = (output_size + workgroup_size - 1) / workgroup_size;
            compute_pass.dispatch_workgroups(num_workgroups as u32, 1, 1);
        }

        queue.submit(std::iter::once(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);

        // Create result tensor
        let device_id = match input.device() {
            crate::Device::Gpu(id) => *id,
            _ => {
                return Err(TensorError::device_error_simple(
                    "Expected GPU device".to_string(),
                ))
            }
        };

        let result_gpu = GpuBuffer::from_wgpu_buffer(
            output_buffer,
            Arc::clone(device),
            Arc::clone(queue),
            crate::Device::Gpu(device_id),
            output_size,
        );

        let mut result = Tensor::from_gpu_buffer(result_gpu, input.shape().clone());
        result.set_requires_grad(input.requires_grad());
        Ok(result)
    } else {
        Err(TensorError::unsupported_operation_simple(
            "GPU batch norm requires GPU tensors".to_string(),
        ))
    }
}

/// GPU Batch Normalization Training Mode Implementation
#[cfg(feature = "gpu")]
fn batch_norm_gpu_training<T>(
    input_gpu: &crate::gpu::buffer::GpuBuffer<T>,
    gamma_gpu: &crate::gpu::buffer::GpuBuffer<T>,
    beta_gpu: &crate::gpu::buffer::GpuBuffer<T>,
    mean_gpu: &crate::gpu::buffer::GpuBuffer<T>,
    var_gpu: &crate::gpu::buffer::GpuBuffer<T>,
    device: &std::sync::Arc<wgpu::Device>,
    queue: &std::sync::Arc<wgpu::Queue>,
    input_shape: &crate::Shape,
    epsilon: T,
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Float
        + FromPrimitive
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    use crate::gpu::buffer::GpuBuffer;
    use crate::tensor::TensorStorage;
    use std::sync::Arc;

    let batch_size = input_shape.dims()[0];
    let channels = input_shape.dims()[1];
    let height = input_shape.dims()[2];
    let width = input_shape.dims()[3];
    let total_elements = batch_size * channels * height * width;

    // Create temporary buffers for batch statistics
    let channel_means_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("batch_norm_channel_means"),
        size: (channels * std::mem::size_of::<T>()) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let channel_vars_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("batch_norm_channel_vars"),
        size: (channels * std::mem::size_of::<T>()) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("batch_norm_training_output"),
        size: (total_elements * std::mem::size_of::<T>()) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Create parameters uniform buffer
    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct BatchNormParams {
        batch_size: u32,
        channels: u32,
        height: u32,
        width: u32,
        epsilon: f32,
    }

    let params = BatchNormParams {
        batch_size: batch_size as u32,
        channels: channels as u32,
        height: height as u32,
        width: width as u32,
        epsilon: epsilon.to_f32().unwrap_or(1e-5),
    };

    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("batch_norm_training_params"),
        contents: bytemuck::cast_slice(&[params]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // Load compute shader
    let shader_source = include_str!("../../gpu/shaders/conv_ops.wgsl");
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("batch_norm_training_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    // Phase 1: Compute means
    {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("batch_norm_mean_layout"),
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
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("batch_norm_mean_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_gpu.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: channel_means_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("batch_norm_mean_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("batch_norm_mean_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("batch_norm_compute_mean"),
            cache: None,
            compilation_options: Default::default(),
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("batch_norm_mean_encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("batch_norm_mean_pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(channels as u32, 1, 1);
        }

        queue.submit(std::iter::once(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);
    }

    // Phase 2: Compute variances
    {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("batch_norm_var_layout"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
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
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("batch_norm_var_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_gpu.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: channel_means_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: channel_vars_buffer.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("batch_norm_var_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("batch_norm_var_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("batch_norm_compute_var"),
            cache: None,
            compilation_options: Default::default(),
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("batch_norm_var_encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("batch_norm_var_pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(channels as u32, 1, 1);
        }

        queue.submit(std::iter::once(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);
    }

    // Phase 3: Apply normalization
    {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("batch_norm_apply_layout"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
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
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("batch_norm_apply_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_gpu.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: channel_means_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: channel_vars_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: gamma_gpu.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: beta_gpu.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("batch_norm_apply_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("batch_norm_apply_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("batch_norm_apply_training"),
            cache: None,
            compilation_options: Default::default(),
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("batch_norm_apply_encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("batch_norm_apply_pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            let workgroup_size = 256;
            let num_workgroups = (total_elements + workgroup_size - 1) / workgroup_size;
            compute_pass.dispatch_workgroups(num_workgroups as u32, 1, 1);
        }

        queue.submit(std::iter::once(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);
    }

    // Phase 4: Update running statistics (exponential moving average)
    // Note: In a real implementation, you might want to make momentum configurable
    let momentum = 0.1f32;

    // Update running mean: running_mean = (1 - momentum) * running_mean + momentum * batch_mean
    // Update running var: running_var = (1 - momentum) * running_var + momentum * batch_var
    {
        let update_stats_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("batch_norm_update_stats_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../gpu/shaders/conv_ops.wgsl").into(),
            ),
        });

        // Create bind group layout for running statistics update
        let update_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("batch_norm_update_stats_layout"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create momentum parameter buffer
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct MomentumParams {
            momentum: f32,
            channels: u32,
            _padding: [u32; 2],
        }

        let momentum_params = MomentumParams {
            momentum,
            channels: channels as u32,
            _padding: [0; 2],
        };

        let momentum_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("batch_norm_momentum_params"),
            contents: bytemuck::cast_slice(&[momentum_params]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let update_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("batch_norm_update_stats_bind_group"),
            layout: &update_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: channel_means_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: channel_vars_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: mean_gpu.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: var_gpu.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: momentum_buffer.as_entire_binding(),
                },
            ],
        });

        let update_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("batch_norm_update_stats_pipeline_layout"),
                bind_group_layouts: &[&update_layout],
                push_constant_ranges: &[],
            });

        let update_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("batch_norm_update_stats_pipeline"),
            layout: Some(&update_pipeline_layout),
            module: &update_stats_shader,
            entry_point: Some("batch_norm_update_running_stats"),
            cache: None,
            compilation_options: Default::default(),
        });

        let mut update_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("batch_norm_update_stats_encoder"),
        });

        {
            let mut compute_pass =
                update_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("batch_norm_update_stats_pass"),
                    timestamp_writes: None,
                });

            compute_pass.set_pipeline(&update_pipeline);
            compute_pass.set_bind_group(0, &update_bind_group, &[]);
            compute_pass.dispatch_workgroups(channels as u32, 1, 1);
        }

        queue.submit(std::iter::once(update_encoder.finish()));
        device.poll(wgpu::Maintain::Wait);
    }

    // Create result tensor
    let device_id = match &crate::Device::Gpu(0) {
        crate::Device::Gpu(id) => *id,
        _ => {
            return Err(TensorError::device_error_simple(
                "Expected GPU device".to_string(),
            ))
        }
    };

    let result_gpu = GpuBuffer::from_wgpu_buffer(
        output_buffer,
        Arc::clone(device),
        Arc::clone(queue),
        crate::Device::Gpu(device_id),
        total_elements,
    );

    let result = Tensor::from_gpu_buffer(result_gpu, input_shape.clone());
    Ok(result)
}
