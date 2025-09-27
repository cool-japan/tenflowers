use crate::gpu::buffer::GpuBuffer;
use crate::gpu_profiler::global_profiler;
use crate::{Device, Result, TensorError};
use std::sync::Arc;
use std::time::Instant;

#[derive(Debug, Clone, Copy)]
pub enum UnaryOp {
    Log,
    Neg,
    Sqrt,
    Abs,
    Exp,
    Sin,
    Cos,
    Tan,
    Tanh,
    ReLU,
    Sigmoid,
    Recip,
    Floor,
    Ceil,
    Round,
}

#[derive(Debug, Clone, Copy)]
pub enum UnaryLogicalOp {
    Not,
}

/// Execute a unary operation on GPU
pub fn execute_unary_op<T>(input_buffer: &GpuBuffer<T>, operation: UnaryOp) -> Result<GpuBuffer<T>>
where
    T: bytemuck::Pod + bytemuck::Zeroable + Clone + Send + Sync + 'static,
{
    let device = &input_buffer.device();
    let queue = &input_buffer.queue();

    // Create output buffer
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("unary_op_output"),
        size: (input_buffer.len() * std::mem::size_of::<T>()) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Create shader module based on data type
    let type_name = std::any::type_name::<T>();
    let shader_source = match type_name {
        "f32" => crate::gpu_include_shader!("unary_ops"),
        "f64" => crate::gpu_include_shader!("unary_ops_f64"),
        "i32" => crate::gpu_include_shader!("unary_ops_i32"),
        "i64" => crate::gpu_include_shader!("unary_ops_i64"),
        "u32" => crate::gpu_include_shader!("unary_ops_u32"),
        "u64" => crate::gpu_include_shader!("unary_ops_u64"),
        _ => crate::gpu_include_shader!("unary_ops"), // fallback to f32
    };

    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("unary_ops_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    // Create bind group layout
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("unary_op_bind_group_layout"),
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
        ],
    });

    // Create bind group
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("unary_op_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buffer.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
        ],
    });

    // Create compute pipeline
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("unary_op_pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let entry_point = get_unary_entry_point(operation, type_name);

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("unary_op_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some(entry_point),
        cache: None,
        compilation_options: Default::default(),
    });

    // Dispatch compute shader with profiling
    let start_time = Instant::now();
    let input_memory = (input_buffer.len() * std::mem::size_of::<T>()) as u64;
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("unary_op_encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("unary_op_pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        let workgroup_size = 64;
        let num_workgroups = (input_buffer.len() + workgroup_size - 1) / workgroup_size;
        compute_pass.dispatch_workgroups(num_workgroups as u32, 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    device.poll(wgpu::Maintain::Wait);

    // Record profiling data
    let execution_time = start_time.elapsed();
    let operation_name = format!("unary_{:?}", operation);
    let _ = global_profiler().record_operation(
        &operation_name,
        input_buffer.device_enum(),
        execution_time,
        input_memory,
    );

    // Create result GpuBuffer
    let device_id = match input_buffer.device_enum() {
        Device::Gpu(id) => id,
        _ => return Err(TensorError::device_mismatch("unary_op", "GPU", "unknown")),
    };

    Ok(GpuBuffer::from_wgpu_buffer(
        output_buffer,
        Arc::clone(&input_buffer.device),
        Arc::clone(&input_buffer.queue),
        Device::Gpu(device_id),
        input_buffer.len(),
    ))
}

fn get_unary_entry_point(operation: UnaryOp, type_name: &str) -> &'static str {
    match (operation, type_name) {
        (UnaryOp::Log, "f32") => "log_op",
        (UnaryOp::Log, "f64") => "log_f64",
        (UnaryOp::Neg, "f32") => "neg_op",
        (UnaryOp::Neg, "f64") => "neg_f64",
        (UnaryOp::Neg, "i32") => "neg_i32",
        (UnaryOp::Neg, "i64") => "neg_i64",
        (UnaryOp::Sqrt, "f32") => "sqrt_op",
        (UnaryOp::Sqrt, "f64") => "sqrt_f64",
        (UnaryOp::Abs, "f32") => "abs_op",
        (UnaryOp::Abs, "f64") => "abs_f64",
        (UnaryOp::Abs, "i32") => "abs_i32",
        (UnaryOp::Abs, "i64") => "abs_i64",
        (UnaryOp::Abs, "u32") => "abs_u32",
        (UnaryOp::Abs, "u64") => "abs_u64",
        (UnaryOp::Exp, "f32") => "exp_op",
        (UnaryOp::Exp, "f64") => "exp_f64",
        (UnaryOp::Sin, "f32") => "sin_op",
        (UnaryOp::Sin, "f64") => "sin_f64",
        (UnaryOp::Cos, "f32") => "cos_op",
        (UnaryOp::Cos, "f64") => "cos_f64",
        (UnaryOp::Tan, "f32") => "tan_op",
        (UnaryOp::Tan, "f64") => "tan_f64",
        (UnaryOp::Tanh, "f32") => "tanh_op",
        (UnaryOp::Tanh, "f64") => "tanh_f64",
        (UnaryOp::ReLU, "f32") => "relu_op",
        (UnaryOp::ReLU, "f64") => "relu_f64",
        (UnaryOp::Sigmoid, "f32") => "sigmoid_op",
        (UnaryOp::Sigmoid, "f64") => "sigmoid_f64",
        (UnaryOp::Recip, "f32") => "recip_op",
        (UnaryOp::Recip, "f64") => "recip_f64",
        (UnaryOp::Floor, "f32") => "floor_op",
        (UnaryOp::Floor, "f64") => "floor_f64",
        (UnaryOp::Ceil, "f32") => "ceil_op",
        (UnaryOp::Ceil, "f64") => "ceil_f64",
        (UnaryOp::Round, "f32") => "round_op",
        (UnaryOp::Round, "f64") => "round_f64",
        // Fallback to f32 operation for unsupported type-operation combinations
        (UnaryOp::Log, _) => "log_op",
        (UnaryOp::Neg, _) => "neg_op",
        (UnaryOp::Sqrt, _) => "sqrt_op",
        (UnaryOp::Abs, _) => "abs_op",
        (UnaryOp::Exp, _) => "exp_op",
        (UnaryOp::Sin, _) => "sin_op",
        (UnaryOp::Cos, _) => "cos_op",
        (UnaryOp::Tan, _) => "tan_op",
        (UnaryOp::Tanh, _) => "tanh_op",
        (UnaryOp::ReLU, _) => "relu_op",
        (UnaryOp::Sigmoid, _) => "sigmoid_op",
        (UnaryOp::Recip, _) => "recip_op",
        (UnaryOp::Floor, _) => "floor_op",
        (UnaryOp::Ceil, _) => "ceil_op",
        (UnaryOp::Round, _) => "round_op",
    }
}
/// Kernel type for unary operations  
pub struct UnaryOpKernel {
    pub operation: UnaryOp,
    pub entry_point: String,
}

impl UnaryOpKernel {
    pub fn new(operation: UnaryOp) -> Self {
        let entry_point = match operation {
            UnaryOp::Neg => "neg_op",
            UnaryOp::Abs => "abs_op",
            UnaryOp::Sqrt => "sqrt_op",
            UnaryOp::Log => "log_op",
            UnaryOp::Exp => "exp_op",
            UnaryOp::Sin => "sin_op",
            UnaryOp::Cos => "cos_op",
            UnaryOp::Tanh => "tanh_op",
            UnaryOp::Sigmoid => "sigmoid_op",
            _ => "generic_op",
        }
        .to_string();

        Self {
            operation,
            entry_point,
        }
    }
}

/// Alias for execute_unary_op for backward compatibility
pub fn gpu_unary_op<T>(input: &GpuBuffer<T>, op: UnaryOp) -> Result<GpuBuffer<T>>
where
    T: bytemuck::Pod + bytemuck::Zeroable + Clone + Send + Sync + 'static,
{
    execute_unary_op(input, op)
}
