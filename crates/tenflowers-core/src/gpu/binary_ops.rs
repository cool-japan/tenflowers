use crate::gpu::buffer::GpuBuffer;
use crate::gpu_profiler::global_profiler;
use crate::{Device, Result, TensorError};
use std::sync::Arc;
use std::time::Instant;

#[derive(Debug, Clone, Copy)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    PReLU,
    Min,
    Max,
    MatMul,
}

/// Get the appropriate shader source for the given type
fn get_binary_op_shader_source<T>() -> &'static str
where
    T: 'static,
{
    let type_name = std::any::type_name::<T>();
    match type_name {
        "f32" => crate::gpu_include_shader!("binary_ops"),
        "f64" => crate::gpu_include_shader!("binary_ops_f64"),
        "i32" => crate::gpu_include_shader!("binary_ops_i32"),
        "i64" => crate::gpu_include_shader!("binary_ops_i64"),
        _ => crate::gpu_include_shader!("binary_ops"), // Default to f32 for unsupported types
    }
}

/// Execute a binary operation on GPU
pub fn execute_binary_op<T>(
    input_a: &GpuBuffer<T>,
    input_b: &GpuBuffer<T>,
    operation: BinaryOp,
    output_len: usize,
) -> Result<GpuBuffer<T>>
where
    T: bytemuck::Pod + bytemuck::Zeroable + Clone + Send + Sync + 'static,
{
    let device = input_a.device();
    let queue = input_a.queue();

    // Create output buffer
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("binary_op_output"),
        size: (output_len * std::mem::size_of::<T>()) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Create shader module with type-specific shader
    let shader_source = get_binary_op_shader_source::<T>();
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("binary_ops_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    // Create bind group layout
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("binary_op_bind_group_layout"),
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
        label: Some("binary_op_bind_group"),
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
        label: Some("binary_op_pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let entry_point = get_binary_entry_point(operation);
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("binary_op_pipeline"),
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
        label: Some("binary_op_encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("binary_op_pass"),
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
    let operation_name = format!("binary_{:?}", operation);
    let _ = global_profiler().record_operation(
        &operation_name,
        input_a.device_enum(),
        execution_time,
        input_memory,
    );

    // Create result GpuBuffer
    let device_id = match input_a.device_enum() {
        Device::Gpu(id) => id,
        _ => return Err(TensorError::device_mismatch("binary_op", "GPU", "unknown")),
    };

    Ok(GpuBuffer::from_wgpu_buffer(
        output_buffer,
        Arc::clone(&input_a.device),
        Arc::clone(&input_a.queue),
        Device::Gpu(device_id),
        output_len,
    ))
}

/// Execute a binary operation with broadcasting on GPU
pub fn execute_binary_op_with_broadcasting<T>(
    input_a: &GpuBuffer<T>,
    input_b: &GpuBuffer<T>,
    operation: BinaryOp,
    shape_a: &[usize],
    shape_b: &[usize],
    broadcast_shape: &[usize],
    output_len: usize,
) -> Result<GpuBuffer<T>>
where
    T: bytemuck::Pod + bytemuck::Zeroable + Clone + Send + Sync + 'static,
{
    let device = input_a.device();
    let queue = input_a.queue();

    // Create output buffer
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("binary_broadcast_output"),
        size: (output_len * std::mem::size_of::<T>()) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Create shape metadata buffer for broadcasting information
    use wgpu::util::DeviceExt;

    // Pack shape metadata: [a_rank, b_rank, output_rank, a_shape..., b_shape..., output_shape...]
    let mut shape_metadata = vec![
        shape_a.len() as u32,
        shape_b.len() as u32,
        broadcast_shape.len() as u32,
    ];

    // Add shape data
    shape_metadata.extend(shape_a.iter().map(|&x| x as u32));
    shape_metadata.extend(shape_b.iter().map(|&x| x as u32));
    shape_metadata.extend(broadcast_shape.iter().map(|&x| x as u32));

    let shape_metadata_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("binary_broadcast_shape_metadata"),
        contents: bytemuck::cast_slice(&shape_metadata),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    // Create shader module
    let shader_source = get_binary_op_shader_source::<T>();
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("binary_broadcast_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    // Create bind group layout with shape metadata
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("binary_broadcast_bind_group_layout"),
        entries: &[
            // Input A
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
            // Input B
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
            // Output
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
            // Shape metadata
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
        label: Some("binary_broadcast_bind_group"),
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
            wgpu::BindGroupEntry {
                binding: 3,
                resource: shape_metadata_buffer.as_entire_binding(),
            },
        ],
    });

    // Create compute pipeline
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("binary_broadcast_pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let entry_point = get_broadcast_entry_point(operation);
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("binary_broadcast_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some(entry_point),
        cache: None,
        compilation_options: Default::default(),
    });

    // Dispatch compute shader
    let start_time = Instant::now();
    let input_memory = ((input_a.len() + input_b.len()) * std::mem::size_of::<T>()) as u64;
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("binary_broadcast_encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("binary_broadcast_pass"),
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
    let operation_name = format!("binary_broadcast_{:?}", operation);
    let _ = global_profiler().record_operation(
        &operation_name,
        input_a.device_enum(),
        execution_time,
        input_memory,
    );

    // Create result GpuBuffer
    let device_id = match input_a.device_enum() {
        Device::Gpu(id) => id,
        _ => return Err(TensorError::device_mismatch("binary_op", "GPU", "unknown")),
    };

    Ok(GpuBuffer::from_wgpu_buffer(
        output_buffer,
        Arc::clone(&input_a.device),
        Arc::clone(&input_a.queue),
        Device::Gpu(device_id),
        output_len,
    ))
}

fn get_binary_entry_point(operation: BinaryOp) -> &'static str {
    match operation {
        BinaryOp::Add => "add_op",
        BinaryOp::Sub => "sub_op",
        BinaryOp::Mul => "mul_op",
        BinaryOp::Div => "div_op",
        BinaryOp::Pow => "pow_op",
        BinaryOp::PReLU => "prelu_op",
        BinaryOp::Min => "min_op",
        BinaryOp::Max => "max_op",
        BinaryOp::MatMul => "matmul_op",
    }
}

fn get_broadcast_entry_point(operation: BinaryOp) -> &'static str {
    match operation {
        BinaryOp::Add => "add_broadcast_op",
        BinaryOp::Sub => "sub_broadcast_op",
        BinaryOp::Mul => "mul_broadcast_op",
        BinaryOp::Div => "div_broadcast_op",
        BinaryOp::Pow => "pow_broadcast_op",
        BinaryOp::PReLU => "prelu_broadcast_op",
        BinaryOp::Min => "min_broadcast_op",
        BinaryOp::Max => "max_broadcast_op",
        BinaryOp::MatMul => "matmul_broadcast_op",
    }
}
/// Kernel type for binary operations
pub struct BinaryOpKernel {
    pub operation: BinaryOp,
    pub entry_point: String,
}

impl BinaryOpKernel {
    pub fn new(operation: BinaryOp) -> Self {
        let entry_point = match operation {
            BinaryOp::Add => "add_op",
            BinaryOp::Sub => "sub_op",
            BinaryOp::Mul => "mul_op",
            BinaryOp::Div => "div_op",
            BinaryOp::Pow => "pow_op",
            BinaryOp::PReLU => "prelu_op",
            BinaryOp::Min => "min_op",
            BinaryOp::Max => "max_op",
            BinaryOp::MatMul => "matmul_op",
        }
        .to_string();

        Self {
            operation,
            entry_point,
        }
    }
}

/// Alias for execute_binary_op for backward compatibility
pub fn gpu_binary_op<T>(
    lhs: &GpuBuffer<T>,
    rhs: &GpuBuffer<T>,
    op: BinaryOp,
) -> Result<GpuBuffer<T>>
where
    T: bytemuck::Pod + bytemuck::Zeroable + Clone + Send + Sync + 'static,
{
    // Use the maximum length for output buffer size
    let output_len = std::cmp::max(lhs.len(), rhs.len());
    execute_binary_op(lhs, rhs, op, output_len)
}
