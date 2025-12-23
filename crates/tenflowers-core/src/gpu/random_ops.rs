//! GPU Random Number Generation Operations
//!
//! This module provides GPU-accelerated random number generation using WGSL compute shaders.

use crate::gpu::buffer::GpuBuffer;
use crate::gpu_profiler::global_profiler;
use crate::{Device, Result, TensorError};
use std::sync::Arc;
use std::time::Instant;

#[derive(Debug, Clone, Copy)]
pub enum RandomOp {
    Normal,
    Uniform,
    StandardNormal,
    Rand,
}

/// Execute random normal distribution generation on GPU
pub fn execute_random_normal<T>(
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    device_enum: Device,
    output_len: usize,
    mean: f32,
    std: f32,
    seed: u64,
) -> Result<GpuBuffer<T>>
where
    T: bytemuck::Pod + bytemuck::Zeroable + Clone + Send + Sync + 'static,
{
    // Create output buffer
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("random_normal_output"),
        size: (output_len * std::mem::size_of::<T>()) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Create parameters buffer: [mean, std, seed_low, seed_high]
    use wgpu::util::DeviceExt;
    let seed_low = (seed & 0xFFFFFFFF) as u32;
    let seed_high = ((seed >> 32) & 0xFFFFFFFF) as u32;

    let params = [
        mean,
        std,
        f32::from_bits(seed_low),
        f32::from_bits(seed_high),
    ];

    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("random_normal_params"),
        contents: bytemuck::cast_slice(&params),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    // Create shader module
    let shader_source = crate::gpu_include_shader!("random_ops");
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("random_normal_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    // Create bind group layout
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("random_normal_bind_group_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
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
        ],
    });

    // Create bind group
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("random_normal_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    // Create compute pipeline
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("random_normal_pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("random_normal_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some("random_normal"),
        cache: None,
        compilation_options: Default::default(),
    });

    // Dispatch compute shader with profiling
    let start_time = Instant::now();
    let output_memory = (output_len * std::mem::size_of::<T>()) as u64;
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("random_normal_encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("random_normal_pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        let workgroup_size = 64;
        let num_workgroups = (output_len + workgroup_size - 1) / workgroup_size;
        compute_pass.dispatch_workgroups(num_workgroups as u32, 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    device.poll(wgpu::Maintain::Wait);

    // Record profiling data
    let execution_time = start_time.elapsed();
    let _ = global_profiler().record_operation(
        "random_normal",
        device_enum,
        execution_time,
        output_memory,
    );

    // Create result GpuBuffer
    let device_id = match device_enum {
        Device::Gpu(id) => id,
        _ => {
            return Err(TensorError::device_mismatch(
                "random_normal",
                "GPU",
                "unknown",
            ))
        }
    };

    Ok(GpuBuffer::from_wgpu_buffer(
        output_buffer,
        device,
        queue,
        Device::Gpu(device_id),
        output_len,
    ))
}

/// Execute random uniform distribution generation on GPU
pub fn execute_random_uniform<T>(
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    device_enum: Device,
    output_len: usize,
    min: f32,
    max: f32,
    seed: u64,
) -> Result<GpuBuffer<T>>
where
    T: bytemuck::Pod + bytemuck::Zeroable + Clone + Send + Sync + 'static,
{
    // Create output buffer
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("random_uniform_output"),
        size: (output_len * std::mem::size_of::<T>()) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Create parameters buffer: [min, max, seed_low, seed_high]
    use wgpu::util::DeviceExt;
    let seed_low = (seed & 0xFFFFFFFF) as u32;
    let seed_high = ((seed >> 32) & 0xFFFFFFFF) as u32;

    let params = [
        min,
        max,
        f32::from_bits(seed_low),
        f32::from_bits(seed_high),
    ];

    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("random_uniform_params"),
        contents: bytemuck::cast_slice(&params),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    // Create shader module
    let shader_source = crate::gpu_include_shader!("random_ops");
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("random_uniform_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    // Create bind group layout
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("random_uniform_bind_group_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
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
        ],
    });

    // Create bind group
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("random_uniform_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    // Create compute pipeline
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("random_uniform_pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("random_uniform_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some("random_uniform"),
        cache: None,
        compilation_options: Default::default(),
    });

    // Dispatch compute shader with profiling
    let start_time = Instant::now();
    let output_memory = (output_len * std::mem::size_of::<T>()) as u64;
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("random_uniform_encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("random_uniform_pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        let workgroup_size = 64;
        let num_workgroups = (output_len + workgroup_size - 1) / workgroup_size;
        compute_pass.dispatch_workgroups(num_workgroups as u32, 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    device.poll(wgpu::Maintain::Wait);

    // Record profiling data
    let execution_time = start_time.elapsed();
    let _ = global_profiler().record_operation(
        "random_uniform",
        device_enum,
        execution_time,
        output_memory,
    );

    // Create result GpuBuffer
    let device_id = match device_enum {
        Device::Gpu(id) => id,
        _ => {
            return Err(TensorError::device_mismatch(
                "random_uniform",
                "GPU",
                "unknown",
            ))
        }
    };

    Ok(GpuBuffer::from_wgpu_buffer(
        output_buffer,
        device,
        queue,
        Device::Gpu(device_id),
        output_len,
    ))
}
