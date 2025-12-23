//! GPU Reduction Operations
//!
//! This module provides GPU-accelerated reduction operations including sum, mean,
//! max, min, product, and other statistical reductions across tensor dimensions.

use super::super::*;
use super::operation_types::ReductionOp;
use crate::Result;

/// Execute a reduction operation on GPU
pub fn execute_reduction_op<T>(
    input: &GpuBuffer<T>,
    op: ReductionOp,
    axes: Option<&[usize]>,
) -> Result<GpuBuffer<T>>
where
    T: bytemuck::Pod + bytemuck::Zeroable + Clone + Send + Sync + 'static,
{
    // Convert axes to i32 for compatibility with execute_axis_reduction_op
    let axes_i32: Option<Vec<i32>> = axes.map(|a| a.iter().map(|&x| x as i32).collect());
    let axes_ref = axes_i32.as_deref();

    // Use shape from buffer or default to 1D shape
    let input_shape = &[input.len()];
    let output_len = 1; // Simple reduction to scalar

    execute_axis_reduction_op(input, op, input_shape, axes_ref, false, output_len)
}

/// Execute a reduction operation along specific axes on GPU
pub fn execute_axis_reduction_op<T>(
    input: &GpuBuffer<T>,
    op: ReductionOp,
    input_shape: &[usize],
    axes: Option<&[i32]>,
    keep_dims: bool,
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
        label: Some("axis_reduction_output"),
        size: (output_len * std::mem::size_of::<T>()) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Calculate reduction parameters
    let input_len = input.len();
    let total_elements = input_shape.iter().product::<usize>();

    // Prepare metadata for the shader
    let mut metadata = vec![input_len as u32, output_len as u32, 0u32, 0u32];

    // Add axis information to metadata
    if let Some(axes_slice) = axes {
        // Pack axis information into metadata
        metadata[2] = axes_slice.len() as u32;
        for (i, &axis) in axes_slice.iter().enumerate() {
            if i < 4 {
                // Limit to first 4 axes for simplicity
                if i == 0 {
                    metadata[3] = axis as u32;
                }
                // Additional axes would need more metadata entries
            }
        }
    }

    let metadata_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("axis_reduction_metadata"),
        contents: bytemuck::cast_slice(&metadata),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    // Load appropriate shader based on operation
    let shader_entry_point = match op {
        ReductionOp::Sum => "sum_reduction",
        ReductionOp::Mean => "mean_reduction",
        ReductionOp::Max => "max_reduction",
        ReductionOp::Min => "min_reduction",
        ReductionOp::Product | ReductionOp::Prod => "product_reduction",
        ReductionOp::ArgMax => "argmax_reduction",
        ReductionOp::ArgMin => "argmin_reduction",
        ReductionOp::All => "all_reduction",
        ReductionOp::Any => "any_reduction",
        ReductionOp::InfNanDetection => "inf_nan_detection",
        ReductionOp::Variance => "variance_reduction",
        ReductionOp::TopK => {
            return Err(crate::TensorError::unsupported_operation_simple(
                "TopK reduction requires specialized implementation".to_string(),
            ))
        }
    };

    let shader_source = include_str!("../shaders/reduction_ops.wgsl");
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("axis_reduction_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    // Create bind group layout
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("axis_reduction_bind_group_layout"),
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
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    // Create compute pipeline
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("axis_reduction_pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("axis_reduction_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some(shader_entry_point),
        cache: None,
        compilation_options: Default::default(),
    });

    // Create bind group
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("axis_reduction_bind_group"),
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
        label: Some("axis_reduction_encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("axis_reduction_pass"),
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
