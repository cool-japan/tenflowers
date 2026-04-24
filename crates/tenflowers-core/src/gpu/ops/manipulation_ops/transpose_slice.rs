//! GPU Tensor Transpose and Slice Operations

use super::super::super::*;
use crate::Result;

/// Execute transpose operation on GPU
pub fn execute_transpose<T>(
    input: &GpuBuffer<T>,
    axes: &[usize],
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
        label: Some("transpose_output"),
        size: (output_len * std::mem::size_of::<T>()) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Calculate output shape
    let mut output_shape = vec![0; input_shape.len()];
    for (i, &axis) in axes.iter().enumerate() {
        output_shape[i] = input_shape[axis];
    }

    // Prepare metadata: [rank, input_shape..., axes..., output_shape...]
    let mut metadata = Vec::new();
    metadata.push(input_shape.len() as u32);
    metadata.extend(input_shape.iter().map(|&x| x as u32));
    metadata.extend(axes.iter().map(|&x| x as u32));
    metadata.extend(output_shape.iter().map(|&x| x as u32));

    let metadata_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("transpose_metadata"),
        contents: bytemuck::cast_slice(&metadata),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    // Load shader
    let shader_source = include_str!("../../shaders/manipulation_ops.wgsl");
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("transpose_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    // Create bind group layout
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("transpose_bind_group_layout"),
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
        label: Some("transpose_pipeline_layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("transpose_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some("transpose_op"),
        cache: None,
        compilation_options: Default::default(),
    });

    // Create bind group
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("transpose_bind_group"),
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
        label: Some("transpose_encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("transpose_pass"),
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

/// Execute slice operation on GPU
pub fn execute_slice<T>(
    input: &GpuBuffer<T>,
    starts: &[usize],
    ends: &[usize],
    steps: &[usize],
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
        label: Some("slice_output"),
        size: (output_len * std::mem::size_of::<T>()) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Calculate output shape
    let mut output_shape = Vec::new();
    for (i, &dim_size) in input_shape.iter().enumerate() {
        let start = starts.get(i).copied().unwrap_or(0);
        let end = ends.get(i).copied().unwrap_or(dim_size);
        let step = steps.get(i).copied().unwrap_or(1);
        output_shape.push((end - start + step - 1) / step);
    }

    // Prepare metadata for shader
    let mut metadata = vec![
        input_shape.len() as u32, // ndim
        output_len as u32,        // total_size
        0u32,                     // pad1
        0u32,                     // pad2
    ];
    metadata.extend(input_shape.iter().map(|&x| x as u32));
    metadata.extend(output_shape.iter().map(|&x| x as u32));
    metadata.extend(starts.iter().map(|&x| x as u32));

    let metadata_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("slice_metadata"),
        contents: bytemuck::cast_slice(&metadata),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    // Load shader and create pipeline
    let shader_source = include_str!("../../shaders/manipulation_ops.wgsl");
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("slice_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("slice_bind_group_layout"),
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
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("slice_pipeline_layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("slice_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some("slice_op"),
        cache: None,
        compilation_options: Default::default(),
    });

    // Create required buffers for the shader
    let input_shape_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("slice_input_shape"),
        contents: bytemuck::cast_slice(&input_shape.iter().map(|&x| x as u32).collect::<Vec<_>>()),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let output_shape_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("slice_output_shape"),
        contents: bytemuck::cast_slice(&output_shape.iter().map(|&x| x as u32).collect::<Vec<_>>()),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let slice_starts_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("slice_starts"),
        contents: bytemuck::cast_slice(&starts.iter().map(|&x| x as u32).collect::<Vec<_>>()),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // Create bind group
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("slice_bind_group"),
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
            wgpu::BindGroupEntry {
                binding: 3,
                resource: input_shape_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: output_shape_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: slice_starts_buffer.as_entire_binding(),
            },
        ],
    });

    // Execute compute shader
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("slice_encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("slice_pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        let workgroup_size = 64;
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
