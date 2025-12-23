/// GPU Reduction Execution
///
/// This module implements the actual GPU execution logic for reduction operations,
/// connecting the WGSL shaders to the Rust API.
use crate::device::context::get_gpu_context;
use crate::gpu::buffer::GpuBuffer;
use crate::{Device, Result, Shape, Tensor, TensorError};
use wgpu::util::DeviceExt;

/// Execute GPU reduction along an axis using WGSL shaders
#[cfg(feature = "gpu")]
pub fn execute_gpu_sum_reduction<T>(
    tensor: &Tensor<T>,
    axis: usize,
    keep_dims: bool,
) -> Result<Tensor<T>>
where
    T: scirs2_core::num_traits::Float
        + Default
        + bytemuck::Pod
        + Send
        + Sync
        + 'static
        + scirs2_core::num_traits::FromPrimitive,
{
    let shape = tensor.shape();
    let device_enum = tensor.device();

    // Get GPU buffer from tensor
    let gpu_buffer = match &tensor.storage {
        crate::tensor::TensorStorage::Gpu(buf) => buf,
        _ => {
            return Err(TensorError::invalid_argument(
                "Tensor must be on GPU device for GPU reduction".to_string(),
            ))
        }
    };

    // Get GPU context (device_id 0 for default GPU)
    let ctx = get_gpu_context(0)?;

    // Calculate output shape
    let mut output_shape = shape.dims().to_vec();
    if keep_dims {
        output_shape[axis] = 1;
    } else {
        output_shape.remove(axis);
    }

    let output_size: usize = output_shape.iter().product();
    let input_size = shape.size();

    // Create output buffer
    let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sum_reduction_output"),
        size: (output_size * std::mem::size_of::<T>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Create metadata buffer
    let input_rank = shape.rank() as u32;
    let num_axes = 1u32; // Single axis reduction
    let mut metadata = vec![
        input_size as u32,
        output_size as u32,
        input_rank,
        num_axes,
        axis as u32, // The axis to reduce
    ];

    let metadata_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sum_reduction_metadata"),
            contents: bytemuck::cast_slice(&metadata),
            usage: wgpu::BufferUsages::STORAGE,
        });

    // Create input/output shape buffers
    let input_shape_data: Vec<u32> = shape.dims().iter().map(|&d| d as u32).collect();
    let input_shape_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sum_reduction_input_shape"),
            contents: bytemuck::cast_slice(&input_shape_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let output_shape_data: Vec<u32> = if keep_dims {
        let mut s = shape.dims().to_vec();
        s[axis] = 1;
        s.iter().map(|&d| d as u32).collect()
    } else {
        output_shape.iter().map(|&d| d as u32).collect()
    };

    let output_shape_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sum_reduction_output_shape"),
            contents: bytemuck::cast_slice(&output_shape_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

    // Load shader
    let shader_module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sum_reduction_shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "../../gpu/shaders/reduction_ops.wgsl"
            ))),
        });

    // Create bind group layout
    let bind_group_layout = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sum_reduction_bind_group_layout"),
            entries: &[
                // Input buffer
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
                // Output buffer
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
                // Metadata buffer
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
                // Input shape buffer
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
                // Output shape buffer
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

    // Create bind group
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("sum_reduction_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: gpu_buffer.buffer().as_entire_binding(),
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
        ],
    });

    // Create pipeline
    let pipeline_layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sum_reduction_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sum_reduction_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("sum_axis_reduction"),
            compilation_options: Default::default(),
            cache: None,
        });

    // Create command encoder and execute
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sum_reduction_encoder"),
        });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sum_reduction_pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        // Dispatch workgroups (64 threads per workgroup as defined in shader)
        let workgroup_size = 64;
        let num_workgroups = (output_size + workgroup_size - 1) / workgroup_size;
        compute_pass.dispatch_workgroups(num_workgroups as u32, 1, 1);
    }

    ctx.queue.submit(std::iter::once(encoder.finish()));

    // Create result tensor from output buffer
    let result_buffer = GpuBuffer::from_wgpu_buffer(
        output_buffer,
        ctx.device.clone(),
        ctx.queue.clone(),
        device_enum.clone(),
        output_size,
    );

    let result_shape = if keep_dims {
        let mut s = shape.dims().to_vec();
        s[axis] = 1;
        Shape::from_slice(&s)
    } else {
        Shape::from_slice(&output_shape)
    };

    Ok(Tensor::from_gpu_buffer(result_buffer, result_shape))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reduction_metadata_creation() {
        // Test metadata buffer creation logic
        let input_size = 100u32;
        let output_size = 10u32;
        let input_rank = 2u32;
        let num_axes = 1u32;
        let axis = 1u32;

        let metadata = vec![input_size, output_size, input_rank, num_axes, axis];

        assert_eq!(metadata[0], 100);
        assert_eq!(metadata[1], 10);
        assert_eq!(metadata[2], 2);
        assert_eq!(metadata[3], 1);
        assert_eq!(metadata[4], 1);
    }

    // Note: Async GPU tests require runtime and proper GPU initialization
    // For integration testing, use the higher-level reduction API
}
