/// GPU Reduction Execution
///
/// This module implements the actual GPU execution logic for reduction operations,
/// connecting the WGSL shaders to the Rust API.
use crate::device::context::get_gpu_context;
use crate::gpu::buffer::GpuBuffer;
use crate::{Device, Result, Shape, Tensor, TensorError};
use wgpu::util::DeviceExt;

/// Reduction operation type for GPU execution
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuReductionOp {
    Sum,
    Mean,
    Max,
    Min,
    Product,
    Variance,
    ArgMax,
    ArgMin,
    All,
    Any,
}

impl GpuReductionOp {
    /// Get the WGSL shader entry point name for this operation
    fn shader_entry_point(&self) -> &'static str {
        match self {
            Self::Sum => "sum_axis_reduction",
            Self::Mean => "mean_axis_reduction",
            Self::Max => "max_axis_reduction",
            Self::Min => "min_axis_reduction",
            Self::Product => "product_axis_reduction",
            Self::Variance => "variance_axis_reduction",
            Self::ArgMax => "argmax_axis_reduction",
            Self::ArgMin => "argmin_axis_reduction",
            Self::All => "all_axis_reduction",
            Self::Any => "any_axis_reduction",
        }
    }

    /// Get operation name for debugging
    fn name(&self) -> &'static str {
        match self {
            Self::Sum => "sum",
            Self::Mean => "mean",
            Self::Max => "max",
            Self::Min => "min",
            Self::Product => "product",
            Self::Variance => "variance",
            Self::ArgMax => "argmax",
            Self::ArgMin => "argmin",
            Self::All => "all",
            Self::Any => "any",
        }
    }
}

/// Execute GPU reduction along an axis using WGSL shaders
#[cfg(feature = "gpu")]
pub fn execute_gpu_reduction<T>(
    tensor: &Tensor<T>,
    axis: usize,
    op: GpuReductionOp,
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
        label: Some(&format!("{}_reduction_output", op.name())),
        size: (output_size * std::mem::size_of::<T>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Create metadata buffer
    let input_rank = shape.rank() as u32;
    let num_axes = 1u32; // Single axis reduction
    let metadata = vec![
        input_size as u32,
        output_size as u32,
        input_rank,
        num_axes,
        axis as u32, // The axis to reduce
    ];

    let metadata_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{}_reduction_metadata", op.name())),
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
            label: Some(&format!("{}_reduction_shader", op.name())),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "../../gpu/shaders/reduction_ops.wgsl"
            ))),
        });

    // Create bind group layout
    let bind_group_layout = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&format!("{}_reduction_bind_group_layout", op.name())),
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
        label: Some(&format!("{}_reduction_bind_group", op.name())),
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
            label: Some(&format!("{}_reduction_pipeline_layout", op.name())),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("{}_reduction_pipeline", op.name())),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some(op.shader_entry_point()),
            compilation_options: Default::default(),
            cache: None,
        });

    // Create command encoder and execute
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(&format!("{}_reduction_encoder", op.name())),
        });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(&format!("{}_reduction_pass", op.name())),
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

/// Convenience function for GPU sum reduction
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
    execute_gpu_reduction(tensor, axis, GpuReductionOp::Sum, keep_dims)
}

/// Convenience function for GPU mean reduction
#[cfg(feature = "gpu")]
pub fn execute_gpu_mean_reduction<T>(
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
    execute_gpu_reduction(tensor, axis, GpuReductionOp::Mean, keep_dims)
}

/// Convenience function for GPU max reduction
#[cfg(feature = "gpu")]
pub fn execute_gpu_max_reduction<T>(
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
    execute_gpu_reduction(tensor, axis, GpuReductionOp::Max, keep_dims)
}

/// Convenience function for GPU min reduction
#[cfg(feature = "gpu")]
pub fn execute_gpu_min_reduction<T>(
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
    execute_gpu_reduction(tensor, axis, GpuReductionOp::Min, keep_dims)
}

/// Convenience function for GPU product reduction
#[cfg(feature = "gpu")]
pub fn execute_gpu_product_reduction<T>(
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
    execute_gpu_reduction(tensor, axis, GpuReductionOp::Product, keep_dims)
}

/// Convenience function for GPU variance reduction
#[cfg(feature = "gpu")]
pub fn execute_gpu_variance_reduction<T>(
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
    execute_gpu_reduction(tensor, axis, GpuReductionOp::Variance, keep_dims)
}

/// Convert gpu_kernels::ReductionOp to GpuReductionOp
impl From<super::gpu_kernels::ReductionOp> for GpuReductionOp {
    fn from(op: super::gpu_kernels::ReductionOp) -> Self {
        match op {
            super::gpu_kernels::ReductionOp::Sum => Self::Sum,
            super::gpu_kernels::ReductionOp::Mean => Self::Mean,
            super::gpu_kernels::ReductionOp::Max => Self::Max,
            super::gpu_kernels::ReductionOp::Min => Self::Min,
            super::gpu_kernels::ReductionOp::Prod => Self::Product,
            super::gpu_kernels::ReductionOp::Variance => Self::Variance,
            super::gpu_kernels::ReductionOp::StdDev => Self::Variance, // StdDev uses variance then sqrt
            super::gpu_kernels::ReductionOp::L1Norm => Self::Sum,      // L1 is sum of abs values
            super::gpu_kernels::ReductionOp::L2Norm => Self::Sum, // L2 is sqrt of sum of squares
            super::gpu_kernels::ReductionOp::Any => Self::Any,
            super::gpu_kernels::ReductionOp::All => Self::All,
        }
    }
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

        let metadata = [input_size, output_size, input_rank, num_axes, axis];

        assert_eq!(metadata[0], 100);
        assert_eq!(metadata[1], 10);
        assert_eq!(metadata[2], 2);
        assert_eq!(metadata[3], 1);
        assert_eq!(metadata[4], 1);
    }

    #[test]
    fn test_gpu_reduction_op_shader_entry_points() {
        assert_eq!(
            GpuReductionOp::Sum.shader_entry_point(),
            "sum_axis_reduction"
        );
        assert_eq!(
            GpuReductionOp::Mean.shader_entry_point(),
            "mean_axis_reduction"
        );
        assert_eq!(
            GpuReductionOp::Max.shader_entry_point(),
            "max_axis_reduction"
        );
        assert_eq!(
            GpuReductionOp::Min.shader_entry_point(),
            "min_axis_reduction"
        );
    }

    #[test]
    fn test_gpu_reduction_op_from_conversion() {
        use super::super::gpu_kernels::ReductionOp;

        assert_eq!(GpuReductionOp::from(ReductionOp::Sum), GpuReductionOp::Sum);
        assert_eq!(
            GpuReductionOp::from(ReductionOp::Mean),
            GpuReductionOp::Mean
        );
        assert_eq!(GpuReductionOp::from(ReductionOp::Max), GpuReductionOp::Max);
        assert_eq!(GpuReductionOp::from(ReductionOp::Min), GpuReductionOp::Min);
    }

    // Note: Async GPU tests require runtime and proper GPU initialization
    // For integration testing, use the higher-level reduction API
}
