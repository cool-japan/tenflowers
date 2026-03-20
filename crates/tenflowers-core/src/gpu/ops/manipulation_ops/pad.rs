//! GPU Tensor Pad Operation

use super::super::super::*;
use crate::Result;

/// Execute pad operation on GPU
pub fn execute_pad<T>(
    input: &GpuBuffer<T>,
    paddings: &[(usize, usize)],
    pad_value: T,
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
        label: Some("pad_output"),
        size: (output_len * std::mem::size_of::<T>()) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Calculate output shape
    let mut output_shape = Vec::new();
    for (i, &dim_size) in input_shape.iter().enumerate() {
        let (pad_before, pad_after) = paddings.get(i).copied().unwrap_or((0, 0));
        output_shape.push(dim_size + pad_before + pad_after);
    }

    // Extract pad_before and pad_after arrays
    let pad_before: Vec<u32> = paddings.iter().map(|(before, _)| *before as u32).collect();
    let pad_after: Vec<u32> = paddings.iter().map(|(_, after)| *after as u32).collect();

    // Suppress unused variable warning for pad_value
    let _ = pad_value;

    // Create uniform data for pad info
    let pad_info = [
        input_shape.len() as u32, // ndim
        output_len as u32,        // total_size
        0u32, // constant_value placeholder - would need proper conversion based on T
        0u32, // pad1
    ];

    let pad_info_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("pad_info"),
        contents: bytemuck::cast_slice(&pad_info),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // Create shape buffers
    let input_shape_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("pad_input_shape"),
        contents: bytemuck::cast_slice(&input_shape.iter().map(|&x| x as u32).collect::<Vec<_>>()),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let output_shape_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("pad_output_shape"),
        contents: bytemuck::cast_slice(&output_shape.iter().map(|&x| x as u32).collect::<Vec<_>>()),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let pad_before_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("pad_before"),
        contents: bytemuck::cast_slice(&pad_before),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let pad_after_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("pad_after"),
        contents: bytemuck::cast_slice(&pad_after),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // Load shader and create pipeline
    let shader_source = include_str!("../../shaders/manipulation_ops.wgsl");
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("pad_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("pad_bind_group_layout"),
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
            wgpu::BindGroupLayoutEntry {
                binding: 6,
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
        label: Some("pad_pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("pad_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some("pad_op"),
        cache: None,
        compilation_options: Default::default(),
    });

    // Create bind group
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("pad_bind_group"),
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
                resource: pad_info_buffer.as_entire_binding(),
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
                resource: pad_before_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: pad_after_buffer.as_entire_binding(),
            },
        ],
    });

    // Execute compute shader
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("pad_encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("pad_pass"),
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
