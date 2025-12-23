//! Layer Normalization Operations
//!
//! This module provides layer normalization operations optimized for transformer
//! architectures and other neural networks that benefit from normalizing across
//! the feature dimension rather than the batch dimension.

use crate::tensor::TensorStorage;
use crate::{Result, Tensor, TensorError};
use scirs2_core::numeric::{Float, FromPrimitive};

/// Layer normalization
/// Input shape: `[..., normalized_shape]`
/// Gamma/beta shapes: `[normalized_shape]`
pub fn layer_norm<T>(
    input: &Tensor<T>,
    gamma: &Tensor<T>,
    beta: &Tensor<T>,
    normalized_shape: &[usize],
    epsilon: T,
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Float
        + FromPrimitive
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    match (&input.storage, &gamma.storage, &beta.storage) {
        (
            TensorStorage::Cpu(input_arr),
            TensorStorage::Cpu(gamma_arr),
            TensorStorage::Cpu(beta_arr),
        ) => {
            let input_shape = input_arr.shape();
            let ndim = input_arr.ndim();

            // Validate normalized_shape
            if normalized_shape.len() > ndim {
                return Err(TensorError::InvalidShape {
                    operation: "layer_norm".to_string(),
                    reason: "normalized_shape has more dimensions than input".to_string(),
                    shape: Some(input_arr.shape().to_vec()),
                    context: None,
                });
            }

            let norm_start = ndim - normalized_shape.len();
            for (i, &dim) in normalized_shape.iter().enumerate() {
                if input_shape[norm_start + i] != dim {
                    return Err(TensorError::ShapeMismatch {
                        operation: "layer_norm".to_string(),
                        expected: format!("normalized_shape {normalized_shape:?}"),
                        got: format!("input shape {input_shape:?}"),
                        context: None,
                    });
                }
            }

            // Validate gamma/beta shapes
            if gamma_arr.shape() != normalized_shape || beta_arr.shape() != normalized_shape {
                return Err(TensorError::ShapeMismatch {
                    operation: "layer_norm".to_string(),
                    expected: format!("gamma/beta shape {normalized_shape:?}"),
                    got: format!(
                        "gamma: {:?}, beta: {:?}",
                        gamma_arr.shape(),
                        beta_arr.shape()
                    ),
                    context: None,
                });
            }

            let mut output = input_arr.clone();
            let total_elements = input_arr.len();
            let norm_size: usize = normalized_shape.iter().product();
            let batch_size = total_elements / norm_size;

            // Reshape for easier processing
            let reshaped_input = input_arr
                .view()
                .into_shape_with_order([batch_size, norm_size])
                .map_err(|e| TensorError::InvalidShape {
                    operation: "layer_norm".to_string(),
                    reason: e.to_string(),
                    shape: None,
                    context: None,
                })?;
            let mut reshaped_output = output
                .view_mut()
                .into_shape_with_order([batch_size, norm_size])
                .map_err(|e| TensorError::InvalidShape {
                    operation: "layer_norm".to_string(),
                    reason: e.to_string(),
                    shape: None,
                    context: None,
                })?;

            let gamma_flat = gamma_arr
                .view()
                .into_shape_with_order([norm_size])
                .map_err(|e| TensorError::InvalidShape {
                    operation: "layer_norm".to_string(),
                    reason: e.to_string(),
                    shape: None,
                    context: None,
                })?;
            let beta_flat = beta_arr
                .view()
                .into_shape_with_order([norm_size])
                .map_err(|e| TensorError::InvalidShape {
                    operation: "layer_norm".to_string(),
                    reason: e.to_string(),
                    shape: None,
                    context: None,
                })?;

            // Normalize each sample
            for i in 0..batch_size {
                // Calculate mean
                let mut sum = T::zero();
                for j in 0..norm_size {
                    sum = sum + reshaped_input[[i, j]];
                }
                let mean = sum / T::from(norm_size).unwrap();

                // Calculate variance
                let mut var_sum = T::zero();
                for j in 0..norm_size {
                    let diff = reshaped_input[[i, j]] - mean;
                    var_sum = var_sum + (diff * diff);
                }
                let variance = var_sum / T::from(norm_size).unwrap();
                let std_dev = (variance + epsilon).sqrt();

                // Normalize and scale
                for j in 0..norm_size {
                    let normalized = (reshaped_input[[i, j]] - mean) / std_dev;
                    reshaped_output[[i, j]] = gamma_flat[[j]] * normalized + beta_flat[[j]];
                }
            }

            Ok(Tensor::from_array(output))
        }
        #[cfg(feature = "gpu")]
        (TensorStorage::Gpu(_), TensorStorage::Gpu(_), TensorStorage::Gpu(_)) => {
            layer_norm_gpu(input, gamma, beta, normalized_shape, epsilon)
        }
        #[cfg(feature = "gpu")]
        _ => Err(TensorError::unsupported_operation_simple(
            "Mixed CPU/GPU layer norm not supported".to_string(),
        )),
    }
}

#[cfg(feature = "gpu")]
fn layer_norm_gpu<T>(
    input: &Tensor<T>,
    gamma: &Tensor<T>,
    beta: &Tensor<T>,
    normalized_shape: &[usize],
    epsilon: T,
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Float
        + FromPrimitive
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    use crate::gpu::buffer::GpuBuffer;
    use crate::tensor::TensorStorage;
    use wgpu::util::DeviceExt;

    if let (
        TensorStorage::Gpu(input_gpu),
        TensorStorage::Gpu(gamma_gpu),
        TensorStorage::Gpu(beta_gpu),
    ) = (&input.storage, &gamma.storage, &beta.storage)
    {
        let device = input_gpu.device();
        let queue = input_gpu.queue();

        let input_shape = input.shape();
        let ndim = input_shape.rank();

        // Validate normalized_shape
        if normalized_shape.len() > ndim {
            return Err(TensorError::InvalidShape {
                operation: "layer_norm".to_string(),
                reason: "normalized_shape has more dimensions than input".to_string(),
                shape: Some(input_shape.dims().to_vec()),
                context: None,
            });
        }

        let norm_start = ndim - normalized_shape.len();
        for (i, &dim) in normalized_shape.iter().enumerate() {
            if input_shape.dims()[norm_start + i] != dim {
                return Err(TensorError::ShapeMismatch {
                    operation: "layer_norm".to_string(),
                    expected: format!("normalized_shape {normalized_shape:?}"),
                    got: format!("input shape {:?}", input_shape.dims()),
                    context: None,
                });
            }
        }

        let total_elements = input_shape.dims().iter().product::<usize>();
        let norm_size: usize = normalized_shape.iter().product();
        let batch_size = total_elements / norm_size;

        // Create output buffer
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("layer_norm_output"),
            size: (total_elements * std::mem::size_of::<T>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create parameters uniform buffer
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct NormalizationParams {
            normalized_size: u32,
            epsilon: f32,
        }

        let params = NormalizationParams {
            normalized_size: norm_size as u32,
            epsilon: epsilon.to_f32().unwrap_or(1e-5),
        };

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("layer_norm_params"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Load compute shader
        let shader_source = include_str!("../../gpu/shaders/normalization_ops.wgsl");
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("layer_norm_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // Choose kernel based on normalized size (fused vs simple)
        let use_fused_kernel = norm_size >= 256;
        let entry_point = if use_fused_kernel {
            "layer_norm" // Fused kernel with shared memory optimization
        } else {
            "layer_norm_simple" // Simple kernel for smaller sizes
        };

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("layer_norm_bind_group_layout"),
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

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("layer_norm_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_gpu.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: gamma_gpu.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: beta_gpu.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Create compute pipeline
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("layer_norm_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("layer_norm_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some(entry_point),
            cache: None,
            compilation_options: Default::default(),
        });

        // Dispatch compute shader
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("layer_norm_encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("layer_norm_pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&compute_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch one workgroup per normalization group
            compute_pass.dispatch_workgroups(batch_size as u32, 1, 1);
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
            input_gpu.device.clone(),
            input_gpu.queue.clone(),
            crate::Device::Gpu(device_id),
            total_elements,
        );

        let mut result = Tensor::from_gpu_buffer(result_gpu, input.shape().clone());
        result.set_requires_grad(input.requires_grad());
        Ok(result)
    } else {
        Err(TensorError::unsupported_operation_simple(
            "GPU layer norm requires GPU tensors".to_string(),
        ))
    }
}
