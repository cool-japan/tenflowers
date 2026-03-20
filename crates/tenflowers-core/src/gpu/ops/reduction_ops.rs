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

    // Compute output shape for axis reductions
    let output_shape: Vec<usize> = if let Some(axes_slice) = axes {
        let mut out_shape = input_shape.to_vec();
        // Sort axes in descending order to avoid index shifting
        let mut sorted_axes: Vec<_> = axes_slice
            .iter()
            .map(|&a| {
                if a < 0 {
                    (input_shape.len() as i32 + a) as usize
                } else {
                    a as usize
                }
            })
            .collect();
        sorted_axes.sort_by(|a, b| b.cmp(a));

        for &axis in &sorted_axes {
            if keep_dims {
                out_shape[axis] = 1;
            } else {
                out_shape.remove(axis);
            }
        }
        if out_shape.is_empty() {
            vec![1] // Scalar result
        } else {
            out_shape
        }
    } else {
        vec![1] // Global reduction to scalar
    };

    // Prepare metadata for the shader
    // Format: [input_size, output_size, input_rank, num_axes, axis0, axis1, ...]
    let mut metadata = vec![
        input_len as u32,
        output_len as u32,
        input_shape.len() as u32, // input_rank
        0u32,                     // num_axes (will be set below)
    ];

    // Add axis information to metadata
    if let Some(axes_slice) = axes {
        metadata[3] = axes_slice.len() as u32; // num_axes
                                               // Add all axes to metadata (up to 4 axes supported)
        for &axis in axes_slice.iter().take(4) {
            let normalized_axis = if axis < 0 {
                (input_shape.len() as i32 + axis) as u32
            } else {
                axis as u32
            };
            metadata.push(normalized_axis);
        }
    }

    let metadata_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("axis_reduction_metadata"),
        contents: bytemuck::cast_slice(&metadata),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    // Create shape buffers for axis reductions
    let input_shape_u32: Vec<u32> = input_shape.iter().map(|&x| x as u32).collect();
    let output_shape_u32: Vec<u32> = output_shape.iter().map(|&x| x as u32).collect();

    let input_shape_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("input_shape"),
        contents: bytemuck::cast_slice(&input_shape_u32),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let output_shape_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("output_shape"),
        contents: bytemuck::cast_slice(&output_shape_u32),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    // Load appropriate shader based on operation and whether it's axis-specific
    // Axis-specific reductions use different entry points that handle multi-dimensional indexing
    let is_axis_reduction = axes.is_some();
    let shader_entry_point = match op {
        ReductionOp::Sum => {
            if is_axis_reduction {
                "sum_axis_reduction"
            } else {
                "sum_reduction"
            }
        }
        ReductionOp::Mean => {
            if is_axis_reduction {
                "mean_axis_reduction"
            } else {
                "mean_reduction"
            }
        }
        ReductionOp::Max => {
            if is_axis_reduction {
                "max_axis_reduction"
            } else {
                "max_reduction"
            }
        }
        ReductionOp::Min => {
            if is_axis_reduction {
                "min_axis_reduction"
            } else {
                "min_reduction"
            }
        }
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

    // Create bind group layout (5 bindings for axis reductions, 3 for global reductions)
    let mut bind_group_entries = vec![
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
    ];

    // Add shape buffer bindings for axis reductions
    if is_axis_reduction {
        bind_group_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 3,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
        bind_group_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 4,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
    }

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("axis_reduction_bind_group_layout"),
        entries: &bind_group_entries,
    });

    // Create compute pipeline
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("axis_reduction_pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        immediate_size: 0,
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
    let mut bind_group_bind_entries = vec![
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
    ];

    // Add shape buffer entries for axis reductions
    if is_axis_reduction {
        bind_group_bind_entries.push(wgpu::BindGroupEntry {
            binding: 3,
            resource: input_shape_buffer.as_entire_binding(),
        });
        bind_group_bind_entries.push(wgpu::BindGroupEntry {
            binding: 4,
            resource: output_shape_buffer.as_entire_binding(),
        });
    }

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("axis_reduction_bind_group"),
        layout: &bind_group_layout,
        entries: &bind_group_bind_entries,
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
