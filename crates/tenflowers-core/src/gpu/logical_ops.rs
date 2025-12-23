//! GPU Logical Operations
//!
//! This module provides GPU-accelerated logical operations using WGSL compute shaders.

use crate::gpu::buffer::GpuBuffer;
use crate::gpu_profiler::global_profiler;
use crate::{Device, Result, TensorError};
use std::sync::Arc;
use std::time::Instant;

#[derive(Debug, Clone, Copy)]
pub enum LogicalOp {
    And,
    Or,
    Xor,
}

#[derive(Debug, Clone, Copy)]
pub enum UnaryLogicalOp {
    Not,
}

/// Execute a binary logical operation on GPU (u32 buffers)
pub fn execute_logical_op<T>(
    input_a: &GpuBuffer<T>,
    input_b: &GpuBuffer<T>,
    operation: LogicalOp,
    output_len: usize,
) -> Result<GpuBuffer<T>>
where
    T: bytemuck::Pod + bytemuck::Zeroable + Clone + Send + Sync + 'static,
{
    let device = input_a.device();
    let queue = input_a.queue();

    // Create output buffer
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("logical_op_output"),
        size: (output_len * std::mem::size_of::<T>()) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Create shader module
    let shader_source = crate::gpu_include_shader!("logical_ops");
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("logical_ops_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    // Create bind group layout
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("logical_op_bind_group_layout"),
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
        ],
    });

    // Create bind group
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("logical_op_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_a.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: input_b.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output_buffer.as_entire_binding(),
            },
        ],
    });

    // Create compute pipeline
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("logical_op_pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let entry_point = get_logical_entry_point(operation);
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("logical_op_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some(entry_point),
        cache: None,
        compilation_options: Default::default(),
    });

    // Dispatch compute shader with profiling
    let start_time = Instant::now();
    let input_memory = ((input_a.len() + input_b.len()) * std::mem::size_of::<T>()) as u64;
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("logical_op_encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("logical_op_pass"),
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
    let operation_name = format!("logical_{:?}", operation);
    let _ = global_profiler().record_operation(
        &operation_name,
        input_a.device_enum(),
        execution_time,
        input_memory,
    );

    // Create result GpuBuffer
    let device_id = match input_a.device_enum() {
        Device::Gpu(id) => id,
        _ => return Err(TensorError::device_mismatch("logical_op", "GPU", "unknown")),
    };

    Ok(GpuBuffer::from_wgpu_buffer(
        output_buffer,
        Arc::clone(&input_a.device),
        Arc::clone(&input_a.queue),
        Device::Gpu(device_id),
        output_len,
    ))
}

/// Execute a unary logical operation on GPU (u32 buffers)
pub fn execute_unary_logical_op<T>(
    input_a: &GpuBuffer<T>,
    operation: UnaryLogicalOp,
    output_len: usize,
) -> Result<GpuBuffer<T>>
where
    T: bytemuck::Pod + bytemuck::Zeroable + Clone + Send + Sync + 'static,
{
    let device = input_a.device();
    let queue = input_a.queue();

    // Create output buffer
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("unary_logical_op_output"),
        size: (output_len * std::mem::size_of::<T>()) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Create shader module
    let shader_source = crate::gpu_include_shader!("logical_ops");
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("unary_logical_ops_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    // Create bind group layout (only need input and output for unary ops)
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("unary_logical_op_bind_group_layout"),
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
                binding: 2,
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

    // Create bind group
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("unary_logical_op_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_a.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output_buffer.as_entire_binding(),
            },
        ],
    });

    // Create compute pipeline
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("unary_logical_op_pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let entry_point = get_unary_logical_entry_point(operation);
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("unary_logical_op_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some(entry_point),
        cache: None,
        compilation_options: Default::default(),
    });

    // Dispatch compute shader with profiling
    let start_time = Instant::now();
    let input_memory = (input_a.len() * std::mem::size_of::<T>()) as u64;
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("unary_logical_op_encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("unary_logical_op_pass"),
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
    let operation_name = format!("unary_logical_{:?}", operation);
    let _ = global_profiler().record_operation(
        &operation_name,
        input_a.device_enum(),
        execution_time,
        input_memory,
    );

    // Create result GpuBuffer
    let device_id = match input_a.device_enum() {
        Device::Gpu(id) => id,
        _ => {
            return Err(TensorError::device_mismatch(
                "unary_logical_op",
                "GPU",
                "unknown",
            ))
        }
    };

    Ok(GpuBuffer::from_wgpu_buffer(
        output_buffer,
        Arc::clone(&input_a.device),
        Arc::clone(&input_a.queue),
        Device::Gpu(device_id),
        output_len,
    ))
}

fn get_logical_entry_point(operation: LogicalOp) -> &'static str {
    match operation {
        LogicalOp::And => "and_op",
        LogicalOp::Or => "or_op",
        LogicalOp::Xor => "xor_op",
    }
}

fn get_unary_logical_entry_point(operation: UnaryLogicalOp) -> &'static str {
    match operation {
        UnaryLogicalOp::Not => "not_op",
    }
}
