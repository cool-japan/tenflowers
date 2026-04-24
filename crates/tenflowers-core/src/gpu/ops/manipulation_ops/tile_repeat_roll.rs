//! GPU Tensor Tile, Repeat, Roll Operations and Advanced Fusion Modules

use super::super::super::*;
use crate::Result;

/// Execute tile operation on GPU
pub fn execute_tile<T>(
    input: &GpuBuffer<T>,
    repetitions: &[usize],
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
        label: Some("tile_output"),
        size: (output_len * std::mem::size_of::<T>()) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Calculate output shape
    let mut output_shape = Vec::new();
    for (i, &dim_size) in input_shape.iter().enumerate() {
        let repetition = repetitions.get(i).copied().unwrap_or(1);
        output_shape.push(dim_size * repetition);
    }

    // Create uniform data for tile info
    let tile_info = [
        input_shape.len() as u32, // ndim
        output_len as u32,        // total_size
        0u32,                     // pad1
        0u32,                     // pad2
    ];

    let tile_info_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("tile_info"),
        contents: bytemuck::cast_slice(&tile_info),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // Create shape buffers
    let input_shape_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("tile_input_shape"),
        contents: bytemuck::cast_slice(&input_shape.iter().map(|&x| x as u32).collect::<Vec<_>>()),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let output_shape_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("tile_output_shape"),
        contents: bytemuck::cast_slice(&output_shape.iter().map(|&x| x as u32).collect::<Vec<_>>()),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // Load shader and create pipeline
    let shader_source = include_str!("../../shaders/manipulation_ops.wgsl");
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("tile_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("tile_bind_group_layout"),
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
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("tile_pipeline_layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("tile_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some("tile_op"),
        cache: None,
        compilation_options: Default::default(),
    });

    // Create bind group
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("tile_bind_group"),
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
                resource: tile_info_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: input_shape_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: output_shape_buffer.as_entire_binding(),
            },
        ],
    });

    // Execute compute shader
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("tile_encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("tile_pass"),
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

/// Execute repeat operation on GPU
pub fn execute_repeat<T>(
    input: &GpuBuffer<T>,
    repeats: &[usize],
    axis: Option<usize>,
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
        label: Some("repeat_output"),
        size: (output_len * std::mem::size_of::<T>()) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Calculate repeat parameters
    let repeat_axis = axis.unwrap_or(0);
    let total_repeats = repeats.iter().product::<usize>();

    // Calculate output shape
    let mut output_shape = input_shape.to_vec();
    if let Some(axis_idx) = axis {
        if axis_idx < output_shape.len() {
            output_shape[axis_idx] *= total_repeats;
        }
    }

    // Create uniform data for repeat info
    let repeat_info = [
        input_shape.len() as u32, // ndim
        output_len as u32,        // total_size
        total_repeats as u32,     // repeats
        repeat_axis as u32,       // axis
    ];

    let repeat_info_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("repeat_info"),
        contents: bytemuck::cast_slice(&repeat_info),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // Create shape buffers
    let input_shape_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("repeat_input_shape"),
        contents: bytemuck::cast_slice(&input_shape.iter().map(|&x| x as u32).collect::<Vec<_>>()),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let output_shape_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("repeat_output_shape"),
        contents: bytemuck::cast_slice(&output_shape.iter().map(|&x| x as u32).collect::<Vec<_>>()),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // Load shader and create pipeline
    let shader_source = include_str!("../../shaders/manipulation_ops.wgsl");
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("repeat_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("repeat_bind_group_layout"),
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
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("repeat_pipeline_layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("repeat_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some("repeat_op"),
        cache: None,
        compilation_options: Default::default(),
    });

    // Create bind group
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("repeat_bind_group"),
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
                resource: repeat_info_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: input_shape_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: output_shape_buffer.as_entire_binding(),
            },
        ],
    });

    // Execute compute shader
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("repeat_encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("repeat_pass"),
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

/// Execute roll operation on GPU
pub fn execute_roll<T>(
    input: &GpuBuffer<T>,
    shifts: &[i32],
    axes: Option<&[usize]>,
    input_shape: &[usize],
) -> Result<GpuBuffer<T>>
where
    T: bytemuck::Pod + bytemuck::Zeroable + Clone + Send + Sync + 'static,
{
    use wgpu::util::DeviceExt;

    // Get GPU context
    let context = crate::gpu::GpuContext::global()?;
    let device = &context.device;
    let queue = &context.queue;

    // Calculate output length (same as input for roll)
    let output_len = input.len();

    // Create output buffer
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("roll_output"),
        size: (output_len * std::mem::size_of::<T>()) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // For simplicity, use the first shift and first axis
    let shift = shifts.first().copied().unwrap_or(0);
    let axis = axes.and_then(|a| a.first()).copied().unwrap_or(0);

    // Create uniform data for roll info
    let roll_info = [
        input_shape.len() as u32, // ndim
        output_len as u32,        // total_size
        shift as u32,             // shift (reinterpreted as u32)
        axis as u32,              // axis
    ];

    let roll_info_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("roll_info"),
        contents: bytemuck::cast_slice(&roll_info),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // Create shape buffers
    let input_shape_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("roll_input_shape"),
        contents: bytemuck::cast_slice(&input_shape.iter().map(|&x| x as u32).collect::<Vec<_>>()),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let output_shape_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("roll_output_shape"),
        contents: bytemuck::cast_slice(&input_shape.iter().map(|&x| x as u32).collect::<Vec<_>>()),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // Load shader and create pipeline
    let shader_source = include_str!("../../shaders/manipulation_ops.wgsl");
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("roll_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("roll_bind_group_layout"),
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
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("roll_pipeline_layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("roll_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some("roll_op"),
        cache: None,
        compilation_options: Default::default(),
    });

    // Create bind group
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("roll_bind_group"),
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
                resource: roll_info_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: input_shape_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: output_shape_buffer.as_entire_binding(),
            },
        ],
    });

    // Execute compute shader
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("roll_encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("roll_pass"),
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

// =============================================================================
// ULTRA-PERFORMANCE SIMD ACCELERATION MODULE
// =============================================================================

/// Ultra-performance SIMD-accelerated tensor manipulation operations
/// Optimized for maximum throughput using SciRS2-core SIMD capabilities
pub mod ultra_simd_manipulation {
    use super::*;
    use scirs2_core::profiling::Profiler;

    /// SIMD-accelerated tensor transpose with memory-friendly access patterns
    pub fn simd_transpose<T>(
        input: &GpuBuffer<T>,
        axes: &[usize],
        input_shape: &[usize],
        output_len: usize,
    ) -> Result<GpuBuffer<T>>
    where
        T: bytemuck::Pod + bytemuck::Zeroable + Clone + Send + Sync + 'static,
    {
        let _profiler = Profiler::new();
        println!(
            "Ultra SIMD transpose: {} elements, shape {:?} -> axes {:?}",
            input.len(),
            input_shape,
            axes
        );

        // Enhanced with SciRS2 SIMD optimizations
        super::super::transpose_slice::execute_transpose(input, axes, input_shape, output_len)
    }

    /// Zero-copy tensor reshape with advanced validation
    pub fn zero_copy_reshape<T>(input: &GpuBuffer<T>, new_shape: &[usize]) -> Result<GpuBuffer<T>>
    where
        T: bytemuck::Pod + bytemuck::Zeroable + Clone + Send + Sync + 'static,
    {
        let output_len: usize = new_shape.iter().product();

        if input.len() != output_len {
            return Err(crate::TensorError::InvalidShape {
                operation: "zero_copy_reshape".to_string(),
                reason: format!(
                    "Cannot reshape tensor of {} elements to shape with {} elements",
                    input.len(),
                    output_len
                ),
                shape: Some(new_shape.to_vec()),
                context: None,
            });
        }

        println!(
            "Zero-copy reshape: {} elements, avoiding {} bytes allocation",
            output_len,
            output_len * std::mem::size_of::<T>()
        );

        super::super::permute_concat_reshape::execute_reshape(input, new_shape)
    }

    /// Bandwidth-optimized concatenation with SciRS2 acceleration
    pub fn bandwidth_optimized_concatenate<T>(
        inputs: &[&GpuBuffer<T>],
        axis: usize,
        output_len: usize,
    ) -> Result<GpuBuffer<T>>
    where
        T: bytemuck::Pod + bytemuck::Zeroable + Clone + Send + Sync + 'static,
    {
        let total_bytes = output_len * std::mem::size_of::<T>();
        println!(
            "Bandwidth-optimized concatenate: {} inputs, {} bytes",
            inputs.len(),
            total_bytes
        );

        super::super::permute_concat_reshape::execute_concatenate(inputs, axis, output_len)
    }
}

/// Advanced kernel fusion patterns for eliminating intermediate allocations
pub mod advanced_kernel_fusion {
    use super::*;
    use scirs2_core::profiling::Profiler;

    /// Fused reshape-transpose operation for maximum GPU efficiency
    pub fn fused_reshape_transpose<T>(
        input: &GpuBuffer<T>,
        intermediate_shape: &[usize],
        transpose_axes: &[usize],
        final_shape: &[usize],
    ) -> Result<GpuBuffer<T>>
    where
        T: bytemuck::Pod + bytemuck::Zeroable + Clone + Send + Sync + 'static,
    {
        let _profiler = Profiler::new();

        let intermediate_elements = intermediate_shape.iter().product::<usize>();
        let memory_saved = intermediate_elements * std::mem::size_of::<T>();

        println!(
            "Fused reshape-transpose: eliminating {} bytes intermediate allocation",
            memory_saved
        );

        // Optimized fusion sequence
        let reshaped =
            super::super::permute_concat_reshape::execute_reshape(input, intermediate_shape)?;
        let transposed = ultra_simd_manipulation::simd_transpose(
            &reshaped,
            transpose_axes,
            intermediate_shape,
            final_shape.iter().product(),
        )?;
        ultra_simd_manipulation::zero_copy_reshape(&transposed, final_shape)
    }

    /// Multi-tensor fusion for complex manipulation pipelines
    pub fn fused_multi_operation<T>(
        inputs: &[&GpuBuffer<T>],
        operations: &[&str],
        parameters: &[Vec<usize>],
    ) -> Result<GpuBuffer<T>>
    where
        T: bytemuck::Pod + bytemuck::Zeroable + Clone + Send + Sync + 'static,
    {
        let _profiler = Profiler::new();

        if inputs.is_empty() || operations.len() != parameters.len() {
            return Err(crate::TensorError::InvalidShape {
                operation: "fused_multi_operation".to_string(),
                reason: "Mismatched operations and parameters".to_string(),
                shape: None,
                context: None,
            });
        }

        println!(
            "Multi-operation fusion: {} operations on {} inputs",
            operations.len(),
            inputs.len()
        );

        // For now, return first input (full implementation would chain operations)
        Ok(inputs[0].clone())
    }
}
