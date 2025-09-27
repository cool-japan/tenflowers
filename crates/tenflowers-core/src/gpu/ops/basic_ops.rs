//! Basic GPU Operations
//!
//! This module provides fundamental GPU operations including unary operations,
//! binary operations, and binary scalar operations. These form the foundation
//! for more complex tensor operations.

use super::super::*;
use crate::Result;
use std::sync::Arc;

/// Execute a unary operation on GPU
pub fn execute_unary_op<T>(input: &GpuBuffer<T>, op: unary_ops::UnaryOp) -> Result<GpuBuffer<T>>
where
    T: bytemuck::Pod + bytemuck::Zeroable + Clone + Send + Sync + 'static,
{
    unary_ops::gpu_unary_op(input, op)
}

/// Execute a binary operation on GPU
pub fn execute_binary_op<T>(
    lhs: &GpuBuffer<T>,
    rhs: &GpuBuffer<T>,
    op: binary_ops::BinaryOp,
) -> Result<GpuBuffer<T>>
where
    T: bytemuck::Pod + bytemuck::Zeroable + Clone + Send + Sync + 'static,
{
    binary_ops::gpu_binary_op(lhs, rhs, op)
}

/// Execute a binary scalar operation on GPU
pub fn execute_binary_scalar_op<T>(
    input: &GpuBuffer<T>,
    scalar: T,
    op: BinaryScalarOp,
) -> Result<GpuBuffer<T>>
where
    T: bytemuck::Pod + bytemuck::Zeroable + Clone + Send + Sync + 'static,
{
    use wgpu::util::DeviceExt;

    // Create output buffer
    let output_buffer = input.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Binary Scalar Output Buffer"),
        size: (input.len() * std::mem::size_of::<T>()) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Create scalar buffer with the scalar value
    let scalar_data = vec![scalar];
    let scalar_buffer = input
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Binary Scalar Value Buffer"),
            contents: bytemuck::cast_slice(&scalar_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

    // Create info buffer with operation parameters
    let info_data = vec![
        input.len() as u32, // input size
        1u32,               // scalar size (always 1)
        op as u32,          // operation type
    ];
    let info_buffer = input
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Binary Scalar Info Buffer"),
            contents: bytemuck::cast_slice(&info_data),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    // Load shader - use binary_ops shader for scalar operations
    let shader_source = include_shader!("binary_ops");
    let shader_module = input
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Binary Scalar Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

    // Create bind group layout
    let bind_group_layout =
        input
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Binary Scalar Bind Group Layout"),
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
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

    // Create bind group
    let bind_group = input.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Binary Scalar Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: scalar_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: info_buffer.as_entire_binding(),
            },
        ],
    });

    // Create compute pipeline
    let compute_pipeline_layout =
        input
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Binary Scalar Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

    let compute_pipeline = input
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Binary Scalar Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &shader_module,
            entry_point: Some("main"),
            cache: None,
            compilation_options: Default::default(),
        });

    // Execute the compute shader
    let mut encoder = input
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Binary Scalar Encoder"),
        });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Binary Scalar Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        // Dispatch with appropriate workgroup size
        let workgroup_size = 64;
        let num_workgroups = (input.len() + workgroup_size - 1) / workgroup_size;
        compute_pass.dispatch_workgroups(num_workgroups as u32, 1, 1);
    }

    input.queue.submit(std::iter::once(encoder.finish()));

    let device_enum = input.device_enum().clone();
    Ok(GpuBuffer::from_wgpu_buffer(
        output_buffer,
        Arc::clone(&input.device),
        Arc::clone(&input.queue),
        device_enum,
        input.len(),
    ))
}

/// Execute a binary operation with broadcasting on GPU
pub fn execute_binary_op_with_broadcasting<T>(
    lhs: &GpuBuffer<T>,
    rhs: &GpuBuffer<T>,
    op: binary_ops::BinaryOp,
    shape_a: &[usize],
    shape_b: &[usize],
) -> Result<GpuBuffer<T>>
where
    T: bytemuck::Pod + bytemuck::Zeroable + Clone + Send + Sync + 'static,
{
    // Calculate broadcast shape - for simplicity, use the larger of the two shapes
    let broadcast_shape = if shape_a.len() >= shape_b.len() {
        shape_a
    } else {
        shape_b
    };
    let output_len = broadcast_shape.iter().product();

    binary_ops::execute_binary_op_with_broadcasting(
        lhs,
        rhs,
        op,
        shape_a,
        shape_b,
        broadcast_shape,
        output_len,
    )
}
