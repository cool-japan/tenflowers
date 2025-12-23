//! GPU Pooling Operations
//!
//! This module provides GPU-accelerated pooling operations including
//! max pooling, average pooling, and fractional pooling for neural networks.

use super::super::*;
use super::operation_types::PoolingOp;
use crate::Result;

/// Execute pooling operation on GPU
pub fn execute_pooling_op<T>(
    input: &GpuBuffer<T>,
    op: PoolingOp,
    kernel_size: &[usize],
    stride: &[usize],
    padding: &[usize],
    input_shape: &[usize],
    output_len: usize,
) -> Result<GpuBuffer<T>>
where
    T: bytemuck::Pod + bytemuck::Zeroable + Clone + Send + Sync + 'static,
{
    use wgpu::util::DeviceExt;

    // Get GPU context
    let context = crate::gpu::GpuContext::global()?;
    let device = &context.device;
    let queue = &context.queue;

    // Create output buffer
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pooling_output"),
        size: (output_len * std::mem::size_of::<T>()) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Prepare metadata for the shader
    let mut metadata = Vec::new();
    metadata.push(input_shape.len() as u32); // rank
    metadata.extend(input_shape.iter().map(|&x| x as u32)); // input shape
    metadata.extend(kernel_size.iter().map(|&x| x as u32)); // kernel size
    metadata.extend(stride.iter().map(|&x| x as u32)); // stride
    metadata.extend(padding.iter().map(|&x| x as u32)); // padding
    metadata.push(output_len as u32); // output length

    let metadata_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("pooling_metadata"),
        contents: bytemuck::cast_slice(&metadata),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    // Load appropriate shader based on operation
    let shader_entry_point = match op {
        PoolingOp::MaxPool2D => "max_pool2d_kernel",
        PoolingOp::AvgPool2D => "avg_pool2d_kernel",
        PoolingOp::GlobalAvgPool => "global_avg_pool2d_kernel",
        PoolingOp::GlobalMaxPool => "global_max_pool2d_kernel",
        _ => {
            return Err(crate::TensorError::unsupported_operation_simple(format!(
                "Pooling operation {:?} not implemented for GPU",
                op
            )))
        }
    };

    let shader_source = include_str!("../shaders/pooling_ops.wgsl");
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("pooling_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    // Create bind group layout matching the existing shader structure
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("pooling_bind_group_layout"),
        entries: &[
            // Binding 0: input
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
            // Binding 1: output
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
            // Binding 2: params (uniform)
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

    // Create compute pipeline
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pooling_pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("pooling_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some(shader_entry_point),
        cache: None,
        compilation_options: Default::default(),
    });

    // Create bind group
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("pooling_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: metadata_buffer.as_entire_binding(),
            },
        ],
    });

    // Execute compute shader
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("pooling_encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("pooling_pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        // Dispatch with 2D/3D workgroup dimensions appropriate for pooling
        match op {
            PoolingOp::GlobalAvgPool | PoolingOp::GlobalMaxPool => {
                // For global pooling, use 1D dispatch
                let num_workgroups = (output_len + 255) / 256;
                compute_pass.dispatch_workgroups(num_workgroups as u32, 1, 1);
            }
            _ => {
                // For spatial pooling, use 2D dispatch (8x8 workgroups)
                let output_height = input_shape.len().saturating_sub(2).max(1);
                let output_width = input_shape.len().saturating_sub(1).max(1);
                let batch_channels =
                    input_shape.first().unwrap_or(&1) * input_shape.get(1).unwrap_or(&1);

                let workgroups_x = (output_width + 7) / 8;
                let workgroups_y = (output_height + 7) / 8;
                let workgroups_z = batch_channels as u32;

                compute_pass.dispatch_workgroups(
                    workgroups_x as u32,
                    workgroups_y as u32,
                    workgroups_z,
                );
            }
        }
    }

    queue.submit(std::iter::once(encoder.finish()));

    // Extract device_id from input buffer
    let device_id = match input.device_enum() {
        Device::Gpu(id) => id,
        _ => 0, // Default for CPU
    };
    // Create GpuBuffer from the result
    Ok(GpuBuffer::from_wgpu_buffer(
        output_buffer,
        context.device.clone(),
        context.queue.clone(),
        Device::Gpu(device_id),
        output_len,
    ))
}

/// Execute fractional pooling operation on GPU
pub fn execute_fractional_pooling_op<T>(
    input: &GpuBuffer<T>,
    op: PoolingOp,
    pooling_ratio: &[f32],
    pseudo_random: bool,
    overlapping: bool,
    input_shape: &[usize],
    output_len: usize,
) -> Result<GpuBuffer<T>>
where
    T: bytemuck::Pod + bytemuck::Zeroable + Clone + Send + Sync + 'static,
{
    use wgpu::util::DeviceExt;

    // Get GPU context
    let context = crate::gpu::GpuContext::global()?;
    let device = &context.device;
    let queue = &context.queue;

    // Create output buffer
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("fractional_pooling_output"),
        size: (output_len * std::mem::size_of::<T>()) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Calculate output shape based on pooling ratio
    let output_height = (input_shape[2] as f32 / pooling_ratio[0]) as u32;
    let output_width = (input_shape[3] as f32 / pooling_ratio[1]) as u32;

    // Create pooling parameters structure
    let pooling_params = [
        input_shape[0] as u32, // batch_size
        input_shape[1] as u32, // channels
        input_shape[2] as u32, // input_height
        input_shape[3] as u32, // input_width
        output_height,         // output_height
        output_width,          // output_width
        0u32,
        0u32, // kernel_height, kernel_width (unused for fractional)
        0u32,
        0u32, // stride_h, stride_w (unused for fractional)
        0u32,
        0u32, // pad_h, pad_w (unused for fractional)
    ];

    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("fractional_pooling_params"),
        contents: bytemuck::cast_slice(&pooling_params),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // Load appropriate fractional pooling shader
    let shader_entry_point = match op {
        PoolingOp::MaxPool2D | PoolingOp::GlobalMaxPool => "fractional_max_pool2d_kernel",
        PoolingOp::AvgPool2D | PoolingOp::GlobalAvgPool => "fractional_adaptive_pool2d",
        _ => {
            return Err(crate::TensorError::unsupported_operation_simple(format!(
                "Fractional pooling operation {:?} not implemented for GPU",
                op
            )))
        }
    };

    let shader_source = include_str!("../shaders/pooling_ops.wgsl");
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("fractional_pooling_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    // Create bind group layout
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("fractional_pooling_bind_group_layout"),
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

    // Create compute pipeline
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("fractional_pooling_pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("fractional_pooling_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some(shader_entry_point),
        cache: None,
        compilation_options: Default::default(),
    });

    // Create bind group
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("fractional_pooling_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    // Execute compute shader
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("fractional_pooling_encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fractional_pooling_pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        // Dispatch with 2D workgroup dimensions (8x8)
        let batch_channels = input_shape[0] * input_shape[1];
        let workgroups_x = (output_width + 7) / 8;
        let workgroups_y = (output_height + 7) / 8;
        let workgroups_z = batch_channels as u32;

        compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
    }

    queue.submit(std::iter::once(encoder.finish()));

    // Extract device_id from input buffer
    let device_id = match input.device_enum() {
        Device::Gpu(id) => id,
        _ => 0, // Default for CPU
    };
    // Create GpuBuffer from the result
    Ok(GpuBuffer::from_wgpu_buffer(
        output_buffer,
        context.device.clone(),
        context.queue.clone(),
        Device::Gpu(device_id),
        output_len,
    ))
}
