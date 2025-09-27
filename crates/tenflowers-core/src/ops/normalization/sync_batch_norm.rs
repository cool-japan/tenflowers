//! Synchronized Batch Normalization Operations
//!
//! This module provides synchronized batch normalization for multi-GPU training.
//! It synchronizes batch statistics across multiple devices to ensure consistent
//! normalization, which is crucial for distributed training scenarios.

use crate::collective::{all_reduce, ReductionOp};
use crate::tensor::TensorStorage;
use crate::{Result, Tensor, TensorError};
use num_traits::{Float, FromPrimitive};

use scirs2_autograd::ndarray::{ArrayD, IxDyn};
#[cfg(feature = "gpu")]
use wgpu::util::DeviceExt;

use super::batch_norm::batch_norm;

/// Synchronized batch normalization for multi-GPU training
/// This synchronizes batch statistics across multiple devices to ensure consistent normalization
/// Input shape: [batch, channels, height, width]
/// Running mean/var shapes: [channels]
/// Gamma/beta shapes: [channels]
#[allow(clippy::too_many_arguments)]
pub fn sync_batch_norm<T>(
    input: &Tensor<T>,
    gamma: &Tensor<T>,
    beta: &Tensor<T>,
    running_mean: &Tensor<T>,
    running_var: &Tensor<T>,
    epsilon: T,
    training: bool,
    momentum: Option<T>,
    group_name: Option<&str>,
) -> Result<(Tensor<T>, Tensor<T>, Tensor<T>)>
where
    T: Clone
        + Default
        + Float
        + FromPrimitive
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable
        + num_traits::Zero
        + num_traits::One
        + std::ops::Add<Output = T>
        + PartialOrd
        + std::ops::Mul<Output = T>,
{
    let input_shape = input.shape();
    if input_shape.rank() != 4 {
        return Err(TensorError::InvalidShape {
            operation: "sync_batch_norm".to_string(),
            reason: "SyncBatchNorm expects 4D input (NCHW format)".to_string(),
            shape: Some(input_shape.dims().to_vec()),
            context: None,
        });
    }

    let batch_size = input_shape.dims()[0];
    let channels = input_shape.dims()[1];
    let height = input_shape.dims()[2];
    let width = input_shape.dims()[3];
    let spatial_size = height * width;

    // Validate parameter shapes
    if gamma.shape().dims() != [channels]
        || beta.shape().dims() != [channels]
        || running_mean.shape().dims() != [channels]
        || running_var.shape().dims() != [channels]
    {
        return Err(TensorError::ShapeMismatch {
            operation: "sync_batch_norm".to_string(),
            expected: format!("parameters shape [{channels}]"),
            got: format!(
                "gamma: {:?}, beta: {:?}, mean: {:?}, var: {:?}",
                gamma.shape().dims(),
                beta.shape().dims(),
                running_mean.shape().dims(),
                running_var.shape().dims()
            ),
            context: None,
        });
    }

    if !training {
        // Inference mode: use running statistics (no synchronization needed)
        let output = batch_norm(
            input,
            gamma,
            beta,
            running_mean,
            running_var,
            epsilon,
            false,
        )?;
        return Ok((output, running_mean.clone(), running_var.clone()));
    }

    // Training mode: compute and synchronize batch statistics
    match (
        &input.storage,
        &gamma.storage,
        &beta.storage,
        &running_mean.storage,
        &running_var.storage,
    ) {
        (
            TensorStorage::Cpu(input_arr),
            TensorStorage::Cpu(gamma_arr),
            TensorStorage::Cpu(beta_arr),
            TensorStorage::Cpu(mean_arr),
            TensorStorage::Cpu(var_arr),
        ) => sync_batch_norm_cpu(
            input_arr,
            gamma_arr,
            beta_arr,
            mean_arr,
            var_arr,
            input,
            gamma,
            beta,
            running_mean,
            running_var,
            batch_size,
            channels,
            height,
            width,
            spatial_size,
            epsilon,
            momentum,
            group_name,
        ),
        #[cfg(feature = "gpu")]
        (
            TensorStorage::Gpu(_),
            TensorStorage::Gpu(_),
            TensorStorage::Gpu(_),
            TensorStorage::Gpu(_),
            TensorStorage::Gpu(_),
        ) => sync_batch_norm_gpu(
            input,
            gamma,
            beta,
            running_mean,
            running_var,
            batch_size,
            channels,
            height,
            width,
            spatial_size,
            epsilon,
            momentum,
            group_name,
        ),
        #[cfg(feature = "gpu")]
        _ => Err(TensorError::unsupported_operation_simple(
            "Mixed CPU/GPU sync batch norm not supported".to_string(),
        )),
    }
}

/// CPU implementation of synchronized batch normalization
#[allow(clippy::too_many_arguments)]
fn sync_batch_norm_cpu<T>(
    input_arr: &ArrayD<T>,
    gamma_arr: &ArrayD<T>,
    beta_arr: &ArrayD<T>,
    mean_arr: &ArrayD<T>,
    var_arr: &ArrayD<T>,
    _input: &Tensor<T>,
    _gamma: &Tensor<T>,
    _beta: &Tensor<T>,
    _running_mean: &Tensor<T>,
    _running_var: &Tensor<T>,
    batch_size: usize,
    channels: usize,
    height: usize,
    width: usize,
    spatial_size: usize,
    epsilon: T,
    momentum: Option<T>,
    group_name: Option<&str>,
) -> Result<(Tensor<T>, Tensor<T>, Tensor<T>)>
where
    T: Clone
        + Default
        + Float
        + FromPrimitive
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable
        + num_traits::Zero
        + num_traits::One
        + std::ops::Add<Output = T>
        + PartialOrd
        + std::ops::Mul<Output = T>,
{
    let momentum = momentum.unwrap_or_else(|| T::from(0.1).unwrap());
    let element_count = batch_size * spatial_size;

    // Step 1: Compute local batch statistics
    let mut local_means = Vec::with_capacity(channels);
    let mut local_vars = Vec::with_capacity(channels);

    for c in 0..channels {
        // Calculate mean for channel c
        let mut sum = T::zero();
        for b in 0..batch_size {
            for h in 0..height {
                for w in 0..width {
                    sum = sum + input_arr[[b, c, h, w]];
                }
            }
        }
        let local_mean = sum / T::from(element_count).unwrap();
        local_means.push(local_mean);

        // Calculate variance for channel c
        let mut var_sum = T::zero();
        for b in 0..batch_size {
            for h in 0..height {
                for w in 0..width {
                    let diff = input_arr[[b, c, h, w]] - local_mean;
                    var_sum = var_sum + (diff * diff);
                }
            }
        }
        let local_var = var_sum / T::from(element_count).unwrap();
        local_vars.push(local_var);
    }

    // Step 2: Create tensors for statistics and synchronize across devices
    let local_mean_tensor = Tensor::from_vec(local_means, &[channels])?;
    let local_var_tensor = Tensor::from_vec(local_vars, &[channels])?;

    // Synchronize means across devices (use mean reduction to get global mean)
    let synced_mean_tensor = all_reduce(&local_mean_tensor, ReductionOp::Mean, group_name)?;

    // For variance, we need to be more careful. We synchronize the sum of squared differences
    // and divide by the total number of elements across all devices
    let element_count_tensor =
        Tensor::from_vec(vec![T::from(element_count).unwrap(); channels], &[channels])?;
    let var_sum_tensor = crate::ops::binary::mul(&local_var_tensor, &element_count_tensor)?;
    let synced_var_sum = all_reduce(&var_sum_tensor, ReductionOp::Sum, group_name)?;
    let total_count_tensor = all_reduce(&element_count_tensor, ReductionOp::Sum, group_name)?;

    // Compute synchronized variance
    let synced_var_tensor = crate::ops::binary::div(&synced_var_sum, &total_count_tensor)?;

    // Step 3: Apply normalization using synchronized statistics
    let mut output = ArrayD::<T>::zeros(IxDyn(&[batch_size, channels, height, width]));

    let synced_means = synced_mean_tensor.as_slice().ok_or_else(|| {
        TensorError::unsupported_operation_simple(
            "Failed to get synchronized means as slice".to_string(),
        )
    })?;
    let synced_vars = synced_var_tensor.as_slice().ok_or_else(|| {
        TensorError::unsupported_operation_simple(
            "Failed to get synchronized variances as slice".to_string(),
        )
    })?;

    for c in 0..channels {
        let mean = synced_means[c];
        let variance = synced_vars[c];
        let std_dev = (variance + epsilon).sqrt();
        let gamma_val = gamma_arr[[c]];
        let beta_val = beta_arr[[c]];

        for b in 0..batch_size {
            for h in 0..height {
                for w in 0..width {
                    let normalized = (input_arr[[b, c, h, w]] - mean) / std_dev;
                    output[[b, c, h, w]] = gamma_val * normalized + beta_val;
                }
            }
        }
    }

    // Step 4: Update running statistics using exponential moving average
    let one_minus_momentum = T::one() - momentum;
    let mut new_running_mean = Vec::with_capacity(channels);
    let mut new_running_var = Vec::with_capacity(channels);

    for c in 0..channels {
        let new_mean = one_minus_momentum * mean_arr[[c]] + momentum * synced_means[c];
        let new_var = one_minus_momentum * var_arr[[c]] + momentum * synced_vars[c];
        new_running_mean.push(new_mean);
        new_running_var.push(new_var);
    }

    let output_tensor = Tensor::from_array(output);
    let updated_mean = Tensor::from_vec(new_running_mean, &[channels])?;
    let updated_var = Tensor::from_vec(new_running_var, &[channels])?;

    Ok((output_tensor, updated_mean, updated_var))
}

/// GPU implementation of synchronized batch normalization
#[cfg(feature = "gpu")]
fn sync_batch_norm_gpu<T>(
    input: &Tensor<T>,
    gamma: &Tensor<T>,
    beta: &Tensor<T>,
    running_mean: &Tensor<T>,
    running_var: &Tensor<T>,
    batch_size: usize,
    channels: usize,
    height: usize,
    width: usize,
    spatial_size: usize,
    epsilon: T,
    momentum: Option<T>,
    group_name: Option<&str>,
) -> Result<(Tensor<T>, Tensor<T>, Tensor<T>)>
where
    T: Clone
        + Default
        + Float
        + FromPrimitive
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable
        + num_traits::Zero
        + num_traits::One
        + std::ops::Add<Output = T>
        + PartialOrd
        + std::ops::Mul<Output = T>,
{
    use crate::gpu::buffer::GpuBuffer;
    use crate::tensor::TensorStorage;
    use wgpu::util::DeviceExt;

    let momentum = momentum.unwrap_or_else(|| T::from(0.1).unwrap());

    if let (
        TensorStorage::Gpu(input_gpu),
        TensorStorage::Gpu(gamma_gpu),
        TensorStorage::Gpu(beta_gpu),
        TensorStorage::Gpu(mean_gpu),
        TensorStorage::Gpu(var_gpu),
    ) = (
        &input.storage,
        &gamma.storage,
        &beta.storage,
        &running_mean.storage,
        &running_var.storage,
    ) {
        let device = input_gpu.device();
        let queue = input_gpu.queue();
        let element_count = batch_size * spatial_size;

        // Step 1: Compute local batch statistics on GPU
        let (local_mean_gpu, local_var_gpu) = compute_batch_stats_gpu::<T>(
            input_gpu,
            &input_gpu.device,
            &input_gpu.queue,
            batch_size,
            channels,
            height,
            width,
            spatial_size,
        )?;

        // Step 2: Convert to tensor and synchronize across devices
        let device_id = match input.device() {
            crate::Device::Gpu(id) => *id,
            _ => {
                return Err(TensorError::device_error_simple(
                    "Expected GPU device".to_string(),
                ))
            }
        };

        let local_mean_tensor =
            Tensor::from_gpu_buffer(local_mean_gpu, crate::Shape::new(vec![channels]));

        let local_var_tensor =
            Tensor::from_gpu_buffer(local_var_gpu, crate::Shape::new(vec![channels]));

        // Synchronize statistics across devices
        let synced_mean_tensor = all_reduce(&local_mean_tensor, ReductionOp::Mean, group_name)?;

        // For variance synchronization, create element count tensor
        let element_count_data = vec![T::from(element_count).unwrap(); channels];
        let element_count_tensor =
            Tensor::from_vec(element_count_data, &[channels])?.to_device(input.device().clone())?;

        let var_sum_tensor = crate::ops::binary::mul(&local_var_tensor, &element_count_tensor)?;
        let synced_var_sum = all_reduce(&var_sum_tensor, ReductionOp::Sum, group_name)?;
        let total_count_tensor = all_reduce(&element_count_tensor, ReductionOp::Sum, group_name)?;
        let synced_var_tensor = crate::ops::binary::div(&synced_var_sum, &total_count_tensor)?;

        // Step 3: Apply normalization using synchronized statistics on GPU
        let output = apply_sync_batch_norm_gpu::<T>(
            input_gpu,
            gamma_gpu,
            beta_gpu,
            &synced_mean_tensor,
            &synced_var_tensor,
            &input_gpu.device,
            &input_gpu.queue,
            batch_size,
            channels,
            height,
            width,
            epsilon,
        )?;

        // Step 4: Update running statistics
        let one_minus_momentum = T::one() - momentum;
        let momentum_tensor = Tensor::from_vec(vec![momentum; channels], &[channels])?
            .to_device(input.device().clone())?;
        let one_minus_momentum_tensor =
            Tensor::from_vec(vec![one_minus_momentum; channels], &[channels])?
                .to_device(input.device().clone())?;

        let momentum_mean = crate::ops::binary::mul(&momentum_tensor, &synced_mean_tensor)?;
        let momentum_var = crate::ops::binary::mul(&momentum_tensor, &synced_var_tensor)?;
        let old_momentum_mean = crate::ops::binary::mul(&one_minus_momentum_tensor, running_mean)?;
        let old_momentum_var = crate::ops::binary::mul(&one_minus_momentum_tensor, running_var)?;

        let updated_mean = crate::ops::binary::add(&old_momentum_mean, &momentum_mean)?;
        let updated_var = crate::ops::binary::add(&old_momentum_var, &momentum_var)?;

        Ok((output, updated_mean, updated_var))
    } else {
        Err(TensorError::unsupported_operation_simple(
            "GPU sync batch norm requires GPU tensors".to_string(),
        ))
    }
}

/// Compute batch statistics on GPU
#[cfg(feature = "gpu")]
fn compute_batch_stats_gpu<T>(
    input_gpu: &crate::gpu::buffer::GpuBuffer<T>,
    device: &std::sync::Arc<wgpu::Device>,
    queue: &std::sync::Arc<wgpu::Queue>,
    batch_size: usize,
    channels: usize,
    height: usize,
    width: usize,
    spatial_size: usize,
) -> Result<(
    crate::gpu::buffer::GpuBuffer<T>,
    crate::gpu::buffer::GpuBuffer<T>,
)>
where
    T: Clone + Default + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    use crate::gpu::buffer::GpuBuffer;

    // Create output buffers for means and variances
    let mean_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sync_batch_norm_means"),
        size: (channels * std::mem::size_of::<T>()) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let var_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sync_batch_norm_vars"),
        size: (channels * std::mem::size_of::<T>()) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Create parameters uniform buffer
    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct BatchStatsParams {
        batch_size: u32,
        channels: u32,
        height: u32,
        width: u32,
        spatial_size: u32,
    }

    let params = BatchStatsParams {
        batch_size: batch_size as u32,
        channels: channels as u32,
        height: height as u32,
        width: width as u32,
        spatial_size: spatial_size as u32,
    };

    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("sync_batch_norm_stats_params"),
        contents: bytemuck::cast_slice(&[params]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // Load compute shader for statistics computation
    let shader_source = include_str!("../../gpu/shaders/normalization_ops.wgsl");
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("sync_batch_norm_stats_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    // Phase 1: Compute means
    {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sync_batch_norm_mean_layout"),
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
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sync_batch_norm_mean_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_gpu.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: mean_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sync_batch_norm_mean_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sync_batch_norm_mean_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("sync_batch_norm_compute_mean"),
            cache: None,
            compilation_options: Default::default(),
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sync_batch_norm_mean_encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sync_batch_norm_mean_pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(channels as u32, 1, 1);
        }

        queue.submit(std::iter::once(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);
    }

    // Phase 2: Compute variances
    {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sync_batch_norm_var_layout"),
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

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sync_batch_norm_var_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_gpu.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: mean_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: var_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sync_batch_norm_var_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sync_batch_norm_var_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("sync_batch_norm_compute_var"),
            cache: None,
            compilation_options: Default::default(),
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sync_batch_norm_var_encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sync_batch_norm_var_pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(channels as u32, 1, 1);
        }

        queue.submit(std::iter::once(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);
    }

    // Create GpuBuffer wrappers
    let device_id = 0; // Default GPU device ID

    let mean_gpu = crate::gpu::buffer::GpuBuffer::from_wgpu_buffer(
        mean_buffer,
        std::sync::Arc::clone(device),
        std::sync::Arc::clone(queue),
        crate::Device::Gpu(device_id),
        channels,
    );

    let var_gpu = crate::gpu::buffer::GpuBuffer::from_wgpu_buffer(
        var_buffer,
        std::sync::Arc::clone(device),
        std::sync::Arc::clone(queue),
        crate::Device::Gpu(device_id),
        channels,
    );

    Ok((mean_gpu, var_gpu))
}

/// Apply synchronized batch normalization on GPU
#[cfg(feature = "gpu")]
fn apply_sync_batch_norm_gpu<T>(
    input_gpu: &crate::gpu::buffer::GpuBuffer<T>,
    gamma_gpu: &crate::gpu::buffer::GpuBuffer<T>,
    beta_gpu: &crate::gpu::buffer::GpuBuffer<T>,
    synced_mean: &Tensor<T>,
    synced_var: &Tensor<T>,
    device: &std::sync::Arc<wgpu::Device>,
    queue: &std::sync::Arc<wgpu::Queue>,
    batch_size: usize,
    channels: usize,
    height: usize,
    width: usize,
    epsilon: T,
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable
        + Float
        + FromPrimitive,
{
    use crate::gpu::buffer::GpuBuffer;
    use crate::tensor::TensorStorage;
    use wgpu::util::DeviceExt;

    let total_elements = batch_size * channels * height * width;

    // Create output buffer
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sync_batch_norm_output"),
        size: (total_elements * std::mem::size_of::<T>()) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Extract GPU buffers from synced statistics
    let synced_mean_gpu = match &synced_mean.storage {
        TensorStorage::Gpu(buf) => buf,
        _ => {
            return Err(TensorError::unsupported_operation_simple(
                "Expected GPU tensor for synced mean".to_string(),
            ))
        }
    };

    let synced_var_gpu = match &synced_var.storage {
        TensorStorage::Gpu(buf) => buf,
        _ => {
            return Err(TensorError::unsupported_operation_simple(
                "Expected GPU tensor for synced var".to_string(),
            ))
        }
    };

    // Create parameters uniform buffer
    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct SyncBatchNormParams {
        batch_size: u32,
        channels: u32,
        height: u32,
        width: u32,
        epsilon: f32,
    }

    let params = SyncBatchNormParams {
        batch_size: batch_size as u32,
        channels: channels as u32,
        height: height as u32,
        width: width as u32,
        epsilon: epsilon.to_f32().unwrap_or(1e-5),
    };

    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("sync_batch_norm_apply_params"),
        contents: bytemuck::cast_slice(&[params]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // Load compute shader
    let shader_source = include_str!("../../gpu/shaders/normalization_ops.wgsl");
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("sync_batch_norm_apply_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    // Create bind group layout
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("sync_batch_norm_apply_layout"),
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
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
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
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 6,
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
        label: Some("sync_batch_norm_apply_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_gpu.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: synced_mean_gpu.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: synced_var_gpu.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: gamma_gpu.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: beta_gpu.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    // Create compute pipeline
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("sync_batch_norm_apply_pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("sync_batch_norm_apply_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some("sync_batch_norm_apply"),
        cache: None,
        compilation_options: Default::default(),
    });

    // Dispatch compute shader
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("sync_batch_norm_apply_encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sync_batch_norm_apply_pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        let workgroup_size = 256;
        let num_workgroups = (total_elements + workgroup_size - 1) / workgroup_size;
        compute_pass.dispatch_workgroups(num_workgroups as u32, 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    device.poll(wgpu::Maintain::Wait);

    // Create result tensor
    let device_id = 0; // Default GPU device ID

    let result_gpu = crate::gpu::buffer::GpuBuffer::from_wgpu_buffer(
        output_buffer,
        std::sync::Arc::clone(device),
        std::sync::Arc::clone(queue),
        crate::Device::Gpu(device_id),
        total_elements,
    );

    let result = Tensor::from_gpu_buffer(
        result_gpu,
        crate::Shape::new(vec![batch_size, channels, height, width]),
    );
    Ok(result)
}
