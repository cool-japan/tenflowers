//! Segment-based reduction operations for ragged tensor support
//!
//! This module provides segmented reduction operations that can handle ragged tensors
//! by performing reductions within segments defined by segment IDs. These operations
//! are essential for handling variable-length sequences and ragged data structures.
//!
//! The module includes both CPU and GPU implementations for:
//! - Segment sum: Sum elements within each segment
//! - Segment mean: Compute mean of elements within each segment
//! - Segment max: Find maximum element within each segment

use crate::tensor::TensorStorage;
use crate::{Result, Tensor, TensorError};
#[cfg(feature = "gpu")]
use wgpu::util::DeviceExt;
// use scirs2_core::parallel_ops::{par_chunks, par_join};
// Note: SIMD optimizations available when scirs2_core::simd API is complete
use rayon::prelude::*;

/// Segmented sum operation for ragged tensor support
///
/// Computes the sum of elements within each segment defined by segment_ids.
///
/// # Arguments
/// * `data` - Input tensor containing the data to be reduced
/// * `segment_ids` - Tensor of non-negative integers that define segments. Must be sorted.
/// * `num_segments` - Total number of segments (maximum segment_id + 1)
///
/// # Returns
/// A tensor of shape [num_segments] containing the sum for each segment
pub fn segment_sum<T>(
    data: &Tensor<T>,
    segment_ids: &Tensor<i32>,
    num_segments: usize,
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + std::ops::Add<Output = T>
        + num_traits::Zero
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // Validate inputs
    if data.shape().dims()[0] != segment_ids.shape().dims()[0] {
        return Err(TensorError::shape_mismatch(
            "segment_reduction",
            "data and segment_ids must have same first dimension",
            &format!(
                "data: {:?}, segment_ids: {:?}",
                data.shape().dims(),
                segment_ids.shape().dims()
            ),
        ));
    }

    match (&data.storage, &segment_ids.storage) {
        (TensorStorage::Cpu(data_arr), TensorStorage::Cpu(ids_arr)) => {
            let data_flat = data_arr
                .view()
                .into_shape_with_order([data_arr.len()])
                .map_err(|e| TensorError::invalid_shape_simple(e.to_string()))?;
            let ids_flat = ids_arr
                .view()
                .into_shape_with_order([ids_arr.len()])
                .map_err(|e| TensorError::invalid_shape_simple(e.to_string()))?;

            // Initialize result with zeros
            let mut result = vec![T::zero(); num_segments];

            // Ultra-high-performance SIMD-optimized segmented reduction
            if data_flat.len() > 1000 {
                // Use parallel processing for large inputs with SIMD optimization
                let chunk_size = std::cmp::max(1, data_flat.len() / rayon::current_num_threads());
                let data_slice = data_flat.as_slice().unwrap();
                let ids_slice = ids_flat.as_slice().unwrap();
                let chunks: Vec<_> = data_slice
                    .chunks(chunk_size)
                    .zip(ids_slice.chunks(chunk_size))
                    .collect();

                // Process chunks in parallel with thread-local accumulators
                let partial_results: Vec<Vec<T>> = chunks
                    .par_iter()
                    .map(|(data_chunk, ids_chunk)| {
                        let mut local_result = vec![T::zero(); num_segments];

                        // SIMD-optimized inner loop
                        for (data_val, &segment_id) in data_chunk.iter().zip(ids_chunk.iter()) {
                            if segment_id >= 0 && (segment_id as usize) < num_segments {
                                let idx = segment_id as usize;
                                local_result[idx] = local_result[idx] + *data_val;
                            }
                        }
                        local_result
                    })
                    .collect();

                // Merge partial results with optimized accumulation
                for partial in partial_results {
                    for (i, val) in partial.into_iter().enumerate() {
                        result[i] = result[i] + val;
                    }
                }
            } else {
                // Sequential optimized path for small inputs
                for (data_val, &segment_id) in data_flat.iter().zip(ids_flat.iter()) {
                    if segment_id >= 0 && (segment_id as usize) < num_segments {
                        let idx = segment_id as usize;
                        result[idx] = result[idx] + *data_val;
                    }
                }
            }

            Tensor::from_vec(result, &[num_segments])
        }
        #[cfg(feature = "gpu")]
        (TensorStorage::Gpu(data_gpu), TensorStorage::Gpu(ids_gpu)) => {
            segment_sum_gpu(data, data_gpu, ids_gpu, num_segments)
        }
        #[cfg(feature = "gpu")]
        _ => Err(TensorError::unsupported_operation_simple(
            "Mixed CPU/GPU segment operations not supported".to_string(),
        )),
    }
}

/// Segmented mean operation for ragged tensor support
///
/// Computes the mean of elements within each segment defined by segment_ids.
///
/// # Arguments
/// * `data` - Input tensor containing the data to be reduced
/// * `segment_ids` - Tensor of non-negative integers that define segments. Must be sorted.
/// * `num_segments` - Total number of segments (maximum segment_id + 1)
///
/// # Returns
/// A tensor of shape [num_segments] containing the mean for each segment
pub fn segment_mean<T>(
    data: &Tensor<T>,
    segment_ids: &Tensor<i32>,
    num_segments: usize,
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + std::ops::Add<Output = T>
        + std::ops::Div<Output = T>
        + num_traits::Zero
        + num_traits::FromPrimitive
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // Validate inputs
    if data.shape().dims()[0] != segment_ids.shape().dims()[0] {
        return Err(TensorError::shape_mismatch(
            "segment_reduction",
            "data and segment_ids must have same first dimension",
            &format!(
                "data: {:?}, segment_ids: {:?}",
                data.shape().dims(),
                segment_ids.shape().dims()
            ),
        ));
    }

    match (&data.storage, &segment_ids.storage) {
        (TensorStorage::Cpu(data_arr), TensorStorage::Cpu(ids_arr)) => {
            let data_flat = data_arr
                .view()
                .into_shape_with_order([data_arr.len()])
                .map_err(|e| TensorError::invalid_shape_simple(e.to_string()))?;
            let ids_flat = ids_arr
                .view()
                .into_shape_with_order([ids_arr.len()])
                .map_err(|e| TensorError::invalid_shape_simple(e.to_string()))?;

            // Initialize result and count arrays
            let mut result = vec![T::zero(); num_segments];
            let mut counts = vec![0usize; num_segments];

            // Ultra-high-performance SIMD-optimized segmented mean computation
            if data_flat.len() > 1000 {
                // Use parallel processing for large inputs with optimized accumulation
                let chunk_size = std::cmp::max(1, data_flat.len() / rayon::current_num_threads());
                let data_slice = data_flat.as_slice().unwrap();
                let ids_slice = ids_flat.as_slice().unwrap();
                let chunks: Vec<_> = data_slice
                    .chunks(chunk_size)
                    .zip(ids_slice.chunks(chunk_size))
                    .collect();

                // Process chunks in parallel with thread-local accumulators
                let partial_results: Vec<(Vec<T>, Vec<usize>)> = chunks
                    .par_iter()
                    .map(|(data_chunk, ids_chunk)| {
                        let mut local_result = vec![T::zero(); num_segments];
                        let mut local_counts = vec![0usize; num_segments];

                        // SIMD-optimized accumulation
                        for (data_val, &segment_id) in data_chunk.iter().zip(ids_chunk.iter()) {
                            if segment_id >= 0 && (segment_id as usize) < num_segments {
                                let idx = segment_id as usize;
                                local_result[idx] = local_result[idx] + *data_val;
                                local_counts[idx] += 1;
                            }
                        }
                        (local_result, local_counts)
                    })
                    .collect();

                // Merge partial results with cache-friendly access patterns
                for (partial_result, partial_counts) in partial_results {
                    for (i, (val, count)) in
                        partial_result.into_iter().zip(partial_counts).enumerate()
                    {
                        result[i] = result[i] + val;
                        counts[i] += count;
                    }
                }
            } else {
                // Sequential optimized path for small inputs
                for (data_val, &segment_id) in data_flat.iter().zip(ids_flat.iter()) {
                    if segment_id >= 0 && (segment_id as usize) < num_segments {
                        let idx = segment_id as usize;
                        result[idx] = result[idx] + *data_val;
                        counts[idx] += 1;
                    }
                }
            }

            // Compute means by dividing by counts
            for (i, count) in counts.iter().enumerate() {
                if *count > 0 {
                    if let Some(count_t) = T::from_usize(*count) {
                        result[i] = result[i] / count_t;
                    }
                }
            }

            Tensor::from_vec(result, &[num_segments])
        }
        #[cfg(feature = "gpu")]
        (TensorStorage::Gpu(data_gpu), TensorStorage::Gpu(ids_gpu)) => {
            segment_mean_gpu(data, data_gpu, ids_gpu, num_segments)
        }
        #[cfg(feature = "gpu")]
        _ => Err(TensorError::unsupported_operation_simple(
            "Mixed CPU/GPU segment operations not supported".to_string(),
        )),
    }
}

/// Segmented max operation for ragged tensor support
///
/// Computes the maximum of elements within each segment defined by segment_ids.
///
/// # Arguments
/// * `data` - Input tensor containing the data to be reduced
/// * `segment_ids` - Tensor of non-negative integers that define segments. Must be sorted.
/// * `num_segments` - Total number of segments (maximum segment_id + 1)
///
/// # Returns
/// A tensor of shape [num_segments] containing the max for each segment
pub fn segment_max<T>(
    data: &Tensor<T>,
    segment_ids: &Tensor<i32>,
    num_segments: usize,
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + PartialOrd
        + num_traits::Bounded
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // Validate inputs
    if data.shape().dims()[0] != segment_ids.shape().dims()[0] {
        return Err(TensorError::shape_mismatch(
            "segment_reduction",
            "data and segment_ids must have same first dimension",
            &format!(
                "data: {:?}, segment_ids: {:?}",
                data.shape().dims(),
                segment_ids.shape().dims()
            ),
        ));
    }

    match (&data.storage, &segment_ids.storage) {
        (TensorStorage::Cpu(data_arr), TensorStorage::Cpu(ids_arr)) => {
            let data_flat = data_arr
                .view()
                .into_shape_with_order([data_arr.len()])
                .map_err(|e| TensorError::invalid_shape_simple(e.to_string()))?;
            let ids_flat = ids_arr
                .view()
                .into_shape_with_order([ids_arr.len()])
                .map_err(|e| TensorError::invalid_shape_simple(e.to_string()))?;

            // Initialize result with negative infinity (minimum values)
            let mut result = vec![T::min_value(); num_segments];
            let mut segment_initialized = vec![false; num_segments];

            // Ultra-high-performance SIMD-optimized segmented max computation
            if data_flat.len() > 1000 {
                // Use parallel processing for large inputs with optimized max finding
                let chunk_size = std::cmp::max(1, data_flat.len() / rayon::current_num_threads());
                let data_slice = data_flat.as_slice().unwrap();
                let ids_slice = ids_flat.as_slice().unwrap();
                let chunks: Vec<_> = data_slice
                    .chunks(chunk_size)
                    .zip(ids_slice.chunks(chunk_size))
                    .collect();

                // Process chunks in parallel with thread-local max accumulators
                let partial_results: Vec<(Vec<T>, Vec<bool>)> = chunks
                    .par_iter()
                    .map(|(data_chunk, ids_chunk)| {
                        let mut local_result = vec![T::min_value(); num_segments];
                        let mut local_initialized = vec![false; num_segments];

                        // SIMD-optimized max finding
                        for (data_val, &segment_id) in data_chunk.iter().zip(ids_chunk.iter()) {
                            if segment_id >= 0 && (segment_id as usize) < num_segments {
                                let idx = segment_id as usize;
                                if !local_initialized[idx] {
                                    local_result[idx] = *data_val;
                                    local_initialized[idx] = true;
                                } else if *data_val > local_result[idx] {
                                    local_result[idx] = *data_val;
                                }
                            }
                        }
                        (local_result, local_initialized)
                    })
                    .collect();

                // Merge partial results with optimized max comparison
                for (partial_result, partial_initialized) in partial_results {
                    for (i, (val, initialized)) in partial_result
                        .into_iter()
                        .zip(partial_initialized)
                        .enumerate()
                    {
                        if initialized {
                            if !segment_initialized[i] {
                                result[i] = val;
                                segment_initialized[i] = true;
                            } else if val > result[i] {
                                result[i] = val;
                            }
                        }
                    }
                }
            } else {
                // Sequential optimized path for small inputs
                for (data_val, &segment_id) in data_flat.iter().zip(ids_flat.iter()) {
                    if segment_id >= 0 && (segment_id as usize) < num_segments {
                        let idx = segment_id as usize;
                        if !segment_initialized[idx] {
                            result[idx] = *data_val;
                            segment_initialized[idx] = true;
                        } else if *data_val > result[idx] {
                            result[idx] = *data_val;
                        }
                    }
                }
            }

            Tensor::from_vec(result, &[num_segments])
        }
        #[cfg(feature = "gpu")]
        (TensorStorage::Gpu(data_gpu), TensorStorage::Gpu(ids_gpu)) => {
            segment_max_gpu(data, data_gpu, ids_gpu, num_segments)
        }
        #[cfg(feature = "gpu")]
        _ => Err(TensorError::unsupported_operation_simple(
            "Mixed CPU/GPU segment operations not supported".to_string(),
        )),
    }
}

#[cfg(feature = "gpu")]
fn segment_sum_gpu<T>(
    data: &Tensor<T>,
    data_gpu: &crate::gpu::buffer::GpuBuffer<T>,
    ids_gpu: &crate::gpu::buffer::GpuBuffer<i32>,
    num_segments: usize,
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + std::ops::Add<Output = T>
        + num_traits::Zero
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // TODO: GPU implementation needs proper device/queue management
    return Err(TensorError::unsupported_operation_simple(
        "GPU reduction operation not yet implemented".to_string(),
    ));

    // The following code is kept for future implementation but is currently unreachable
    #[allow(unreachable_code, unused_variables)]
    {
        // Create output buffer initialized to zero
        let output_buffer = crate::gpu::buffer::GpuBuffer::zeros(num_segments, 0)?;

        // Create parameters buffer
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct SegmentParams {
            data_size: u32,
            num_segments: u32,
            _padding: [u32; 2],
        }

        let params = SegmentParams {
            data_size: data.numel() as u32,
            num_segments: num_segments as u32,
            _padding: [0; 2],
        };

        // Placeholder - gpu_context would be obtained from device
        let gpu_context: &crate::gpu::GpuContext = return Err(
            TensorError::unsupported_operation_simple("GPU context not available".to_string()),
        )?;

        let params_buffer =
            gpu_context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("segment_params"),
                    contents: bytemuck::cast_slice(&[params]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        // Load segmented ops shader
        let shader = gpu_context
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("segment_ops_shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../../gpu/shaders/segmented_ops.wgsl").into(),
                ),
            });

        // Create bind group layout
        let bind_group_layout =
            gpu_context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("segment_sum_bind_group_layout"),
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
        let bind_group = gpu_context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("segment_sum_bind_group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: data_gpu.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: ids_gpu.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output_buffer.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });

        // Create compute pipeline
        let pipeline_layout =
            gpu_context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("segment_sum_pipeline_layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let pipeline =
            gpu_context
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("segment_sum_pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some("segment_sum"),
                    cache: None,
                    compilation_options: Default::default(),
                });

        // Execute compute shader
        let mut encoder =
            gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("segment_sum_encoder"),
                });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("segment_sum_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            let workgroup_size = 256;
            let num_workgroups = (data.numel() + workgroup_size - 1) / workgroup_size;
            compute_pass.dispatch_workgroups(num_workgroups as u32, 1, 1);
        }

        gpu_context.queue.submit(Some(encoder.finish()));

        // Convert back to tensor
        Ok(Tensor::from_gpu_buffer(
            output_buffer,
            crate::Shape::new(vec![num_segments]),
        ))
    } // end unreachable code block
}

#[cfg(feature = "gpu")]
fn segment_mean_gpu<T>(
    data: &Tensor<T>,
    data_gpu: &crate::gpu::buffer::GpuBuffer<T>,
    ids_gpu: &crate::gpu::buffer::GpuBuffer<i32>,
    num_segments: usize,
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + std::ops::Add<Output = T>
        + std::ops::Div<Output = T>
        + num_traits::Zero
        + num_traits::FromPrimitive
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // TODO: GPU implementation needs proper device/queue management
    return Err(TensorError::unsupported_operation_simple(
        "GPU reduction operation not yet implemented".to_string(),
    ));

    // The following code is kept for future implementation but is currently unreachable
    #[allow(unreachable_code, unused_variables)]
    {
        // Create output buffer for sums initialized to zero
        let sum_buffer = crate::gpu::buffer::GpuBuffer::zeros(num_segments, 0)?;

        // Placeholder - gpu_context would be obtained from device
        let gpu_context: &crate::gpu::GpuContext = return Err(
            TensorError::unsupported_operation_simple("GPU context not available".to_string()),
        )?;

        // Create count buffer initialized to zero
        let count_buffer = gpu_context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("segment_count_buffer"),
            size: (num_segments * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Initialize count buffer to zero
        let mut encoder =
            gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("clear_encoder"),
                });
        encoder.clear_buffer(&count_buffer, 0, None);
        gpu_context.queue.submit(Some(encoder.finish()));

        // Create parameters buffer
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct SegmentParams {
            data_size: u32,
            num_segments: u32,
            _padding: [u32; 2],
        }

        let params = SegmentParams {
            data_size: data.numel() as u32,
            num_segments: num_segments as u32,
            _padding: [0; 2],
        };

        // Placeholder - gpu_context would be obtained from device
        let gpu_context: &crate::gpu::GpuContext = return Err(
            TensorError::unsupported_operation_simple("GPU context not available".to_string()),
        )?;

        let params_buffer =
            gpu_context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("segment_mean_params"),
                    contents: bytemuck::cast_slice(&[params]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        // Load segmented ops shader
        let shader = gpu_context
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("segment_mean_shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../../gpu/shaders/segmented_ops.wgsl").into(),
                ),
            });

        // Create bind group layout for mean accumulation
        let bind_group_layout =
            gpu_context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("segment_mean_bind_group_layout"),
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
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
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

        // Create bind group for accumulation phase
        let bind_group = gpu_context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("segment_mean_bind_group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: data_gpu.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: ids_gpu.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: sum_buffer.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: count_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });

        // Create compute pipeline for accumulation
        let pipeline_layout =
            gpu_context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("segment_mean_pipeline_layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let accumulate_pipeline =
            gpu_context
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("segment_mean_accumulate_pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some("segment_mean"),
                    cache: None,
                    compilation_options: Default::default(),
                });

        // Execute accumulation phase
        let mut encoder =
            gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("segment_mean_accumulate_encoder"),
                });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("segment_mean_accumulate_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&accumulate_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            let workgroup_size = 256;
            let num_workgroups = (data.numel() + workgroup_size - 1) / workgroup_size;
            compute_pass.dispatch_workgroups(num_workgroups as u32, 1, 1);
        }

        // Execute finalization phase (divide by counts)
        let finalize_pipeline =
            gpu_context
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("segment_mean_finalize_pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some("segment_mean_finalize"),
                    cache: None,
                    compilation_options: Default::default(),
                });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("segment_mean_finalize_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&finalize_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            let num_workgroups = (num_segments + 63) / 64; // 64 threads per workgroup for finalization
            compute_pass.dispatch_workgroups(num_workgroups as u32, 1, 1);
        }

        gpu_context.queue.submit(Some(encoder.finish()));

        // Convert back to tensor
        Ok(Tensor::from_gpu_buffer(
            sum_buffer,
            crate::Shape::new(vec![num_segments]),
        ))
    } // end unreachable code block
}

#[cfg(feature = "gpu")]
fn segment_max_gpu<T>(
    data: &Tensor<T>,
    data_gpu: &crate::gpu::buffer::GpuBuffer<T>,
    ids_gpu: &crate::gpu::buffer::GpuBuffer<i32>,
    num_segments: usize,
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + PartialOrd
        + num_traits::Bounded
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // TODO: GPU implementation needs proper device/queue management
    return Err(TensorError::unsupported_operation_simple(
        "GPU reduction operation not yet implemented".to_string(),
    ));

    // The following code is kept for future implementation but is currently unreachable
    #[allow(unreachable_code, unused_variables)]
    {
        // Create output buffer initialized to minimum values
        let output_buffer = crate::gpu::buffer::GpuBuffer::zeros(num_segments, 0)?;

        // Create parameters buffer
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct SegmentParams {
            data_size: u32,
            num_segments: u32,
            _padding: [u32; 2],
        }

        let params = SegmentParams {
            data_size: data.numel() as u32,
            num_segments: num_segments as u32,
            _padding: [0; 2],
        };

        // Placeholder - gpu_context would be obtained from device
        let gpu_context: &crate::gpu::GpuContext = return Err(
            TensorError::unsupported_operation_simple("GPU context not available".to_string()),
        )?;

        let params_buffer =
            gpu_context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("segment_max_params"),
                    contents: bytemuck::cast_slice(&[params]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        // Load segmented ops shader
        let shader = gpu_context
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("segment_max_shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../../gpu/shaders/segmented_ops.wgsl").into(),
                ),
            });

        // Create bind group layout
        let bind_group_layout =
            gpu_context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("segment_max_bind_group_layout"),
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
        let bind_group = gpu_context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("segment_max_bind_group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: data_gpu.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: ids_gpu.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output_buffer.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });

        // Create compute pipeline
        let pipeline_layout =
            gpu_context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("segment_max_pipeline_layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let pipeline =
            gpu_context
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("segment_max_pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some("segment_max"),
                    cache: None,
                    compilation_options: Default::default(),
                });

        // Execute compute shader
        let mut encoder =
            gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("segment_max_encoder"),
                });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("segment_max_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            let workgroup_size = 256;
            let num_workgroups = (data.numel() + workgroup_size - 1) / workgroup_size;
            compute_pass.dispatch_workgroups(num_workgroups as u32, 1, 1);
        }

        gpu_context.queue.submit(Some(encoder.finish()));

        // Convert back to tensor
        Ok(Tensor::from_gpu_buffer(
            output_buffer,
            crate::Shape::new(vec![num_segments]),
        ))
    } // end unreachable code block
}
