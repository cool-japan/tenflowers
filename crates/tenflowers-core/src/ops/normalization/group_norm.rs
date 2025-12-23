//! Group Normalization Operations
//!
//! This module provides group normalization operations designed for stable training
//! with small batch sizes. Group normalization divides channels into groups and
//! normalizes within each group, making it independent of batch size.

use crate::tensor::TensorStorage;
use crate::{Result, Tensor, TensorError};
use scirs2_core::numeric::{Float, FromPrimitive};

/// Group normalization
/// Input shape: `[batch, channels, height, width]`
/// Gamma/beta shapes: `[channels]`
pub fn group_norm<T>(
    input: &Tensor<T>,
    gamma: &Tensor<T>,
    beta: &Tensor<T>,
    num_groups: usize,
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
    match (&input.storage, &gamma.storage, &beta.storage) {
        (
            TensorStorage::Cpu(input_arr),
            TensorStorage::Cpu(gamma_arr),
            TensorStorage::Cpu(beta_arr),
        ) => {
            let input_shape = input_arr.shape();
            let ndim = input_arr.ndim();

            if ndim < 2 {
                return Err(TensorError::InvalidShape {
                    operation: "group_norm".to_string(),
                    reason: "GroupNorm expects at least 2D input (batch, channels, ...)"
                        .to_string(),
                    shape: Some(input_arr.shape().to_vec()),
                    context: None,
                });
            }

            let batch_size = input_shape[0];
            let channels = input_shape[1];

            // Validate num_groups
            if channels % num_groups != 0 {
                return Err(TensorError::invalid_argument(format!(
                    "channels {channels} must be divisible by num_groups {num_groups}"
                )));
            }

            let channels_per_group = channels / num_groups;

            // Validate gamma/beta shapes
            if gamma_arr.shape() != [channels] || beta_arr.shape() != [channels] {
                return Err(TensorError::ShapeMismatch {
                    operation: "group_norm".to_string(),
                    expected: format!("gamma/beta shape [{channels}]"),
                    got: format!(
                        "gamma: {:?}, beta: {:?}",
                        gamma_arr.shape(),
                        beta_arr.shape()
                    ),
                    context: None,
                });
            }

            let mut output = input_arr.clone();

            // Calculate spatial size (everything after batch and channels)
            let spatial_size: usize = input_shape[2..].iter().product();

            // Process each batch
            for b in 0..batch_size {
                // Process each group
                for g in 0..num_groups {
                    let start_channel = g * channels_per_group;
                    let end_channel = start_channel + channels_per_group;

                    // Calculate mean for this group
                    let mut sum = T::zero();
                    let mut count = 0;

                    for c in start_channel..end_channel {
                        for s in 0..spatial_size {
                            let mut full_idx = vec![b, c];

                            // Add spatial indices in correct order
                            let mut s_temp = s;
                            let mut spatial_idx = Vec::new();
                            for d in (2..ndim).rev() {
                                spatial_idx.push(s_temp % input_shape[d]);
                                s_temp /= input_shape[d];
                            }
                            spatial_idx.reverse();
                            full_idx.extend(spatial_idx);

                            sum = sum + input_arr[full_idx.as_slice()];
                            count += 1;
                        }
                    }

                    let mean = sum / T::from(count).unwrap();

                    // Calculate variance for this group
                    let mut var_sum = T::zero();
                    for c in start_channel..end_channel {
                        for s in 0..spatial_size {
                            let mut full_idx = vec![b, c];

                            // Add spatial indices in correct order
                            let mut s_temp = s;
                            let mut spatial_idx = Vec::new();
                            for d in (2..ndim).rev() {
                                spatial_idx.push(s_temp % input_shape[d]);
                                s_temp /= input_shape[d];
                            }
                            spatial_idx.reverse();
                            full_idx.extend(spatial_idx);

                            let diff = input_arr[full_idx.as_slice()] - mean;
                            var_sum = var_sum + (diff * diff);
                        }
                    }

                    let variance = var_sum / T::from(count).unwrap();
                    let std_dev = (variance + epsilon).sqrt();

                    // Normalize and scale for this group
                    for c in start_channel..end_channel {
                        let gamma_val = gamma_arr[[c]];
                        let beta_val = beta_arr[[c]];

                        for s in 0..spatial_size {
                            let mut full_idx = vec![b, c];

                            // Add spatial indices in correct order
                            let mut s_temp = s;
                            let mut spatial_idx = Vec::new();
                            for d in (2..ndim).rev() {
                                spatial_idx.push(s_temp % input_shape[d]);
                                s_temp /= input_shape[d];
                            }
                            spatial_idx.reverse();
                            full_idx.extend(spatial_idx);

                            let normalized = (input_arr[full_idx.as_slice()] - mean) / std_dev;
                            output[full_idx.as_slice()] = gamma_val * normalized + beta_val;
                        }
                    }
                }
            }

            Ok(Tensor::from_array(output))
        }
        #[cfg(feature = "gpu")]
        (TensorStorage::Gpu(_), TensorStorage::Gpu(_), TensorStorage::Gpu(_)) => {
            group_norm_gpu(input, gamma, beta, num_groups, epsilon)
        }
        #[cfg(feature = "gpu")]
        _ => Err(TensorError::unsupported_operation_simple(
            "Mixed CPU/GPU group norm not supported".to_string(),
        )),
    }
}

#[cfg(feature = "gpu")]
fn group_norm_gpu<T>(
    input: &Tensor<T>,
    gamma: &Tensor<T>,
    beta: &Tensor<T>,
    num_groups: usize,
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
    use wgpu::util::DeviceExt;

    if let (
        TensorStorage::Gpu(input_gpu),
        TensorStorage::Gpu(gamma_gpu),
        TensorStorage::Gpu(beta_gpu),
    ) = (&input.storage, &gamma.storage, &beta.storage)
    {
        let device = input_gpu.device();
        let queue = input_gpu.queue();

        let input_shape = input.shape();
        let ndim = input_shape.rank();

        if ndim < 2 {
            return Err(TensorError::InvalidShape {
                operation: "group_norm".to_string(),
                reason: "GroupNorm expects at least 2D input (batch, channels, ...)".to_string(),
                shape: Some(input_shape.dims().to_vec()),
                context: None,
            });
        }

        let batch_size = input_shape.dims()[0];
        let channels = input_shape.dims()[1];

        // Validate num_groups
        if channels % num_groups != 0 {
            return Err(TensorError::invalid_argument(format!(
                "channels {channels} must be divisible by num_groups {num_groups}"
            )));
        }

        let channels_per_group = channels / num_groups;

        // Calculate spatial size (everything after batch and channels)
        let spatial_size: usize = input_shape.dims()[2..].iter().product();

        // Create output buffer
        let total_elements = input_shape.dims().iter().product::<usize>();
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("group_norm_output"),
            size: (total_elements * std::mem::size_of::<T>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create parameters uniform buffer
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct GroupNormParams {
            batch_size: u32,
            num_channels: u32,
            num_groups: u32,
            spatial_size: u32,
            epsilon: f32,
        }

        let params = GroupNormParams {
            batch_size: batch_size as u32,
            num_channels: channels as u32,
            num_groups: num_groups as u32,
            spatial_size: spatial_size as u32,
            epsilon: epsilon.to_f32().unwrap_or(1e-5),
        };

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("group_norm_params"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Load compute shader
        let shader_source = include_str!("../../gpu/shaders/normalization_ops.wgsl");
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("group_norm_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("group_norm_bind_group_layout"),
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

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("group_norm_bind_group"),
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
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Create compute pipeline
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("group_norm_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("group_norm_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("group_norm"),
            cache: None,
            compilation_options: Default::default(),
        });

        // Dispatch compute shader
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("group_norm_encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("group_norm_pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&compute_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // The shader expects workgroup_id.x = batch_idx, workgroup_id.y = group_idx
            compute_pass.dispatch_workgroups(batch_size as u32, num_groups as u32, 1);
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
            input_gpu.device.clone(),
            input_gpu.queue.clone(),
            crate::Device::Gpu(device_id),
            total_elements,
        );

        let mut result = Tensor::from_gpu_buffer(result_gpu, input.shape().clone());
        result.set_requires_grad(input.requires_grad());
        Ok(result)
    } else {
        Err(TensorError::unsupported_operation_simple(
            "GPU group norm requires GPU tensors".to_string(),
        ))
    }
}
