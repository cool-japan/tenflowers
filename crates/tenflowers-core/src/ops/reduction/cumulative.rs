//! Cumulative reduction operations
//!
//! This module provides cumulative reduction operations such as cumulative sum and
//! cumulative product. These operations compute running totals along specified axes
//! while preserving the original tensor shape.
//!
//! # Operations
//! - `cumsum`: Cumulative sum along an axis
//! - `cumprod`: Cumulative product along an axis
//!
//! # GPU Support
//! GPU implementations are available when the "gpu" feature is enabled, providing
//! highly optimized compute shader-based implementations for large arrays.

use crate::shape_error_taxonomy::{validate_reduction_axis, ShapeErrorUtils};
use crate::tensor::TensorStorage;
use crate::{Result, Tensor, TensorError};
use std::sync::Arc;

/// Cumulative sum along a specified axis
///
/// Computes the cumulative sum of elements along the specified axis.
/// The output has the same shape as the input tensor.
///
/// # Arguments
/// * `x` - Input tensor
/// * `axis` - Axis along which to compute cumulative sum. If None, uses the last axis.
///
/// # Returns
/// A tensor with the same shape as input containing cumulative sums
///
/// # Examples
/// ```
/// use tenflowers_core::{Tensor, ops::cumsum};
///
/// let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
/// let result = cumsum(&tensor, Some(1)).unwrap();
/// // Result: [[1.0, 3.0], [3.0, 7.0]]
/// ```
pub fn cumsum<T>(x: &Tensor<T>, axis: Option<i32>) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + std::ops::Add<Output = T>
        + scirs2_core::num_traits::Zero
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    let shape = x.shape().dims();

    // Default to last axis if not specified
    let axis_raw = axis.unwrap_or(-1);

    // Validate and normalize axis using standardized validation
    let axis = validate_reduction_axis("cumsum", axis_raw as isize, x.shape())?;

    match &x.storage {
        TensorStorage::Cpu(array) => {
            let mut result_data = vec![T::zero(); array.len()];

            // Copy input data first
            if let Some(input_slice) = x.as_slice() {
                for (i, value) in input_slice.iter().enumerate() {
                    result_data[i] = *value;
                }
            } else {
                return Err(TensorError::unsupported_operation_simple(
                    "Failed to get tensor data".to_string(),
                ));
            }

            // Compute strides for the target axis
            let axis_size = shape[axis];
            let mut stride = 1;
            for &dim in &shape[axis + 1..] {
                stride *= dim;
            }

            let outer_size = shape[..axis].iter().product::<usize>();

            // Perform cumulative sum along the specified axis
            for outer_idx in 0..outer_size {
                for inner_idx in 0..stride {
                    let base_idx = outer_idx * axis_size * stride + inner_idx;

                    for i in 1..axis_size {
                        let curr_idx = base_idx + i * stride;
                        let prev_idx = base_idx + (i - 1) * stride;
                        result_data[curr_idx] = result_data[curr_idx] + result_data[prev_idx];
                    }
                }
            }

            Tensor::from_vec(result_data, shape)
        }
        #[cfg(feature = "gpu")]
        TensorStorage::Gpu(gpu_buffer) => gpu_cumsum(x, gpu_buffer, axis, shape),
    }
}

/// Cumulative product along a specified axis
///
/// Computes the cumulative product of elements along the specified axis.
/// The output has the same shape as the input tensor.
///
/// # Arguments
/// * `x` - Input tensor
/// * `axis` - Axis along which to compute cumulative product. If None, uses the last axis.
///
/// # Returns
/// A tensor with the same shape as input containing cumulative products
///
/// # Examples
/// ```
/// use tenflowers_core::{Tensor, ops::cumprod};
///
/// let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
/// let result = cumprod(&tensor, Some(1)).unwrap();
/// // Result: [[1.0, 2.0], [3.0, 12.0]]
/// ```
pub fn cumprod<T>(x: &Tensor<T>, axis: Option<i32>) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + std::ops::Mul<Output = T>
        + scirs2_core::num_traits::One
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    let shape = x.shape().dims();

    // Default to last axis if not specified
    let axis_raw = axis.unwrap_or(-1);

    // Validate and normalize axis using standardized validation
    let axis = validate_reduction_axis("cumprod", axis_raw as isize, x.shape())?;

    match &x.storage {
        TensorStorage::Cpu(array) => {
            let mut result_data = vec![T::one(); array.len()];

            // Copy input data first
            if let Some(input_slice) = x.as_slice() {
                for (i, value) in input_slice.iter().enumerate() {
                    result_data[i] = *value;
                }
            } else {
                return Err(TensorError::unsupported_operation_simple(
                    "Failed to get tensor data".to_string(),
                ));
            }

            // Compute strides for the target axis
            let axis_size = shape[axis];
            let mut stride = 1;
            for &dim in &shape[axis + 1..] {
                stride *= dim;
            }

            let outer_size = shape[..axis].iter().product::<usize>();

            // Perform cumulative product along the specified axis
            for outer_idx in 0..outer_size {
                for inner_idx in 0..stride {
                    let base_idx = outer_idx * axis_size * stride + inner_idx;

                    for i in 1..axis_size {
                        let curr_idx = base_idx + i * stride;
                        let prev_idx = base_idx + (i - 1) * stride;
                        result_data[curr_idx] = result_data[curr_idx] * result_data[prev_idx];
                    }
                }
            }

            Tensor::from_vec(result_data, shape)
        }
        #[cfg(feature = "gpu")]
        TensorStorage::Gpu(gpu_buffer) => gpu_cumprod(x, gpu_buffer, axis, shape),
    }
}

#[cfg(feature = "gpu")]
fn gpu_cumsum<T>(
    input: &Tensor<T>,
    gpu_buffer: &crate::gpu::buffer::GpuBuffer<T>,
    axis: usize,
    shape: &[usize],
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + std::ops::Add<Output = T>
        + scirs2_core::num_traits::Zero
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    use crate::gpu::buffer::GpuBuffer;
    use crate::tensor::TensorStorage;
    use wgpu::util::DeviceExt;

    let device = gpu_buffer.device();
    let queue = gpu_buffer.queue();
    let total_size = shape.iter().product::<usize>();

    // Calculate stride and axis size for the scan operation
    let axis_size = shape[axis];
    let mut stride = 1;
    for &dim in &shape[axis + 1..] {
        stride *= dim;
    }

    // Create output buffer
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cumsum_output"),
        size: (total_size * std::mem::size_of::<T>()) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Create temporary buffer for large scans
    let temp_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cumsum_temp"),
        size: ((total_size / 256 + 1) * std::mem::size_of::<T>()) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Create parameters uniform buffer
    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct ScanParams {
        size: u32,
        axis: u32,
        stride: u32,
        axis_size: u32,
    }

    let params = ScanParams {
        size: total_size as u32,
        axis: axis as u32,
        stride: stride as u32,
        axis_size: axis_size as u32,
    };

    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("cumsum_params"),
        contents: bytemuck::cast_slice(&[params]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // Load compute shader
    let shader_source = include_str!("../../gpu/shaders/scan_ops.wgsl");
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("scan_ops_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    // Create bind group layout
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("cumsum_bind_group_layout"),
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
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    // Create bind group
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cumsum_bind_group"),
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
                resource: temp_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    // Select kernel based on array size and axis
    let entry_point = if axis_size <= 256 && total_size <= 256 * 256 {
        "cumsum_simple"
    } else if shape.len() > 1 {
        "cumsum_axis"
    } else {
        "cumsum_optimized"
    };

    // Create compute pipeline
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("cumsum_pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("cumsum_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some(entry_point),
        cache: None,
        compilation_options: Default::default(),
    });

    // Dispatch compute shader
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("cumsum_encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cumsum_pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        // Calculate workgroup size based on kernel type
        let workgroup_count = match entry_point {
            "cumsum_simple" => (total_size + 255) / 256,
            "cumsum_axis" => (total_size + 63) / 64,
            "cumsum_optimized" => (total_size + 511) / 512, // 2 elements per thread
            _ => (total_size + 255) / 256,
        };

        compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    device.poll(wgpu::Maintain::Wait);

    // Create result tensor
    let device_id = match input.device() {
        crate::Device::Gpu(id) => *id,
        _ => {
            return Err(TensorError::device_error_simple(
                "Expected GPU device".to_string(),
            ))
        }
    };

    let result_gpu = GpuBuffer::from_wgpu_buffer(
        output_buffer,
        Arc::clone(&gpu_buffer.device),
        Arc::clone(&gpu_buffer.queue),
        crate::Device::Gpu(device_id),
        total_size,
    );

    let mut result = Tensor::from_gpu_buffer(result_gpu, input.shape().clone());
    result.set_requires_grad(input.requires_grad());
    Ok(result)
}

#[cfg(feature = "gpu")]
fn gpu_cumprod<T>(
    input: &Tensor<T>,
    gpu_buffer: &crate::gpu::buffer::GpuBuffer<T>,
    axis: usize,
    shape: &[usize],
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + std::ops::Mul<Output = T>
        + scirs2_core::num_traits::One
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    use crate::gpu::buffer::GpuBuffer;
    use crate::tensor::TensorStorage;
    use wgpu::util::DeviceExt;

    let device = gpu_buffer.device();
    let queue = gpu_buffer.queue();
    let total_size = shape.iter().product::<usize>();

    // Calculate stride and axis size for the scan operation
    let axis_size = shape[axis];
    let mut stride = 1;
    for &dim in &shape[axis + 1..] {
        stride *= dim;
    }

    // Create output buffer
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cumprod_output"),
        size: (total_size * std::mem::size_of::<T>()) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Create temporary buffer for large scans
    let temp_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cumprod_temp"),
        size: ((total_size / 256 + 1) * std::mem::size_of::<T>()) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Create parameters uniform buffer
    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct ScanParams {
        size: u32,
        axis: u32,
        stride: u32,
        axis_size: u32,
    }

    let params = ScanParams {
        size: total_size as u32,
        axis: axis as u32,
        stride: stride as u32,
        axis_size: axis_size as u32,
    };

    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("cumprod_params"),
        contents: bytemuck::cast_slice(&[params]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // Load compute shader
    let shader_source = include_str!("../../gpu/shaders/scan_ops.wgsl");
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("scan_ops_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    // Create bind group layout (reuse the same layout as cumsum)
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("cumprod_bind_group_layout"),
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
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    // Create bind group
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cumprod_bind_group"),
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
                resource: temp_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    // Select kernel based on array size and axis
    let entry_point = if axis_size <= 256 && total_size <= 256 * 256 {
        "cumprod_simple"
    } else if shape.len() > 1 {
        "cumprod_axis"
    } else {
        "cumprod_simple" // Default to simple for 1D
    };

    // Create compute pipeline
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("cumprod_pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("cumprod_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some(entry_point),
        cache: None,
        compilation_options: Default::default(),
    });

    // Dispatch compute shader
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("cumprod_encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cumprod_pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        // Calculate workgroup size based on kernel type
        let workgroup_count = match entry_point {
            "cumprod_simple" => (total_size + 255) / 256,
            "cumprod_axis" => (total_size + 63) / 64,
            _ => (total_size + 255) / 256,
        };

        compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    device.poll(wgpu::Maintain::Wait);

    // Create result tensor
    let device_id = match input.device() {
        crate::Device::Gpu(id) => *id,
        _ => {
            return Err(TensorError::device_error_simple(
                "Expected GPU device".to_string(),
            ))
        }
    };

    let result_gpu = GpuBuffer::from_wgpu_buffer(
        output_buffer,
        Arc::clone(&gpu_buffer.device),
        Arc::clone(&gpu_buffer.queue),
        crate::Device::Gpu(device_id),
        total_size,
    );

    let mut result = Tensor::from_gpu_buffer(result_gpu, input.shape().clone());
    result.set_requires_grad(input.requires_grad());
    Ok(result)
}
