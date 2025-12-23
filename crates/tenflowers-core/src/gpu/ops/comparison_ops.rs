//! GPU Comparison Operations
//!
//! This module provides GPU-accelerated comparison operations including
//! equality, inequality, greater than, less than comparisons between tensors.

use super::super::*;
use super::operation_types::ComparisonOp;
use crate::Result;

/// Execute a comparison operation on GPU
/// Returns a GpuBuffer<u32> where 0 represents false and 1 represents true
pub fn execute_comparison_op<T>(
    lhs: &GpuBuffer<T>,
    rhs: &GpuBuffer<T>,
    op: ComparisonOp,
    output_len: usize,
) -> Result<GpuBuffer<u32>>
where
    T: bytemuck::Pod + bytemuck::Zeroable + Clone + Send + Sync + 'static,
{
    use wgpu::util::DeviceExt;

    // Get GPU context
    let context = crate::gpu::GpuContext::global()?;
    let device = &context.device;
    let queue = &context.queue;

    // Create output buffer for boolean results (u32: 0 or 1)
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("comparison_output"),
        size: (output_len * std::mem::size_of::<u32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Create metadata buffer
    let metadata = [
        lhs.len() as u32,
        rhs.len() as u32,
        output_len as u32,
        op as u32,
    ];
    let metadata_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("comparison_metadata"),
        contents: bytemuck::cast_slice(&metadata),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    // Load appropriate shader based on operation (f32 version for now)
    let shader_entry_point = match op {
        ComparisonOp::Eq => "eq_f32",
        ComparisonOp::Ne => "ne_f32",
        ComparisonOp::Lt => "lt_f32",
        ComparisonOp::Le => "le_f32",
        ComparisonOp::Gt => "gt_f32",
        ComparisonOp::Ge => "ge_f32",
    };

    let shader_source = include_str!("../shaders/comparison_ops.wgsl");
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("comparison_ops_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    // Create bind group layout matching the existing shader
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("comparison_bind_group_layout"),
        entries: &[
            // Binding 0: input_a_f32
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
            // Binding 1: input_b_f32
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
            // Binding 2: output_bool
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

    // Create compute pipeline
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("comparison_pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("comparison_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some(shader_entry_point),
        cache: None,
        compilation_options: Default::default(),
    });

    // Create bind group
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("comparison_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: lhs.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: rhs.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output_buffer.as_entire_binding(),
            },
        ],
    });

    // Execute compute shader
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("comparison_encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("comparison_pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        // Dispatch with workgroup size optimized for GPU architecture
        let workgroup_size = 256;
        let num_workgroups = (output_len + workgroup_size - 1) / workgroup_size;
        compute_pass.dispatch_workgroups(num_workgroups as u32, 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));

    // Extract device_id from input buffer
    let device_id = match lhs.device_enum() {
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

/// Execute a comparison operation on GPU with broadcasting support
/// Returns a GpuBuffer<u32> where 0 represents false and 1 represents true
pub fn execute_comparison_op_with_broadcasting<T>(
    lhs: &GpuBuffer<T>,
    rhs: &GpuBuffer<T>,
    op: ComparisonOp,
    lhs_shape: &[usize],
    rhs_shape: &[usize],
    output_shape: &[usize],
    output_len: usize,
) -> Result<GpuBuffer<u32>>
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
        label: Some("comparison_broadcast_output"),
        size: (output_len * std::mem::size_of::<u32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Create shape metadata buffer
    // Format: [a_rank, b_rank, output_rank, a_shape..., b_shape..., output_shape...]
    let a_rank = lhs_shape.len() as u32;
    let b_rank = rhs_shape.len() as u32;
    let output_rank = output_shape.len() as u32;

    let mut shape_metadata = vec![a_rank, b_rank, output_rank];
    shape_metadata.extend(lhs_shape.iter().map(|&s| s as u32));
    shape_metadata.extend(rhs_shape.iter().map(|&s| s as u32));
    shape_metadata.extend(output_shape.iter().map(|&s| s as u32));

    let shape_metadata_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("comparison_broadcast_shape_metadata"),
        contents: bytemuck::cast_slice(&shape_metadata),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    // Determine shader entry point based on operation type
    let type_name = std::any::type_name::<T>();
    let shader_entry_point = if type_name.contains("f32") {
        match op {
            ComparisonOp::Eq => "eq_f32_broadcast",
            ComparisonOp::Ne => "ne_f32_broadcast",
            ComparisonOp::Lt => "lt_f32_broadcast",
            ComparisonOp::Le => "le_f32_broadcast",
            ComparisonOp::Gt => "gt_f32_broadcast",
            ComparisonOp::Ge => "ge_f32_broadcast",
        }
    } else if type_name.contains("f64") {
        match op {
            ComparisonOp::Eq => "eq_f64_broadcast",
            ComparisonOp::Ne => "ne_f64_broadcast",
            ComparisonOp::Lt => "lt_f64_broadcast",
            ComparisonOp::Le => "le_f64_broadcast",
            ComparisonOp::Gt => "gt_f64_broadcast",
            ComparisonOp::Ge => "ge_f64_broadcast",
        }
    } else if type_name.contains("i32") {
        match op {
            ComparisonOp::Eq => "eq_i32_broadcast",
            ComparisonOp::Ne => "ne_i32_broadcast",
            ComparisonOp::Lt => "lt_i32_broadcast",
            ComparisonOp::Le => "le_i32_broadcast",
            ComparisonOp::Gt => "gt_i32_broadcast",
            ComparisonOp::Ge => "ge_i32_broadcast",
        }
    } else {
        // Default to f32
        match op {
            ComparisonOp::Eq => "eq_f32_broadcast",
            ComparisonOp::Ne => "ne_f32_broadcast",
            ComparisonOp::Lt => "lt_f32_broadcast",
            ComparisonOp::Le => "le_f32_broadcast",
            ComparisonOp::Gt => "gt_f32_broadcast",
            ComparisonOp::Ge => "ge_f32_broadcast",
        }
    };

    let shader_source = include_str!("../shaders/comparison_ops.wgsl");
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("comparison_broadcast_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    // Create bind group layout
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("comparison_broadcast_bind_group_layout"),
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
        label: Some("comparison_broadcast_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: lhs.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: rhs.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: shape_metadata_buffer.as_entire_binding(),
            },
        ],
    });

    // Create compute pipeline
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("comparison_broadcast_pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("comparison_broadcast_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some(shader_entry_point),
        cache: None,
        compilation_options: Default::default(),
    });

    // Execute compute shader
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("comparison_broadcast_encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("comparison_broadcast_pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        let workgroup_size = 64;
        let num_workgroups = (output_len + workgroup_size - 1) / workgroup_size;
        compute_pass.dispatch_workgroups(num_workgroups as u32, 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));

    // Extract device_id
    let device_id = match lhs.device_enum() {
        Device::Gpu(id) => id,
        _ => 0,
    };

    Ok(GpuBuffer::from_wgpu_buffer(
        output_buffer,
        context.device.clone(),
        context.queue.clone(),
        Device::Gpu(device_id),
        output_len,
    ))
}
