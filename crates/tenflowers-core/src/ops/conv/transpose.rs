//! Transpose convolution (deconvolution) operations
//!
//! This module provides transpose convolution implementations for both CPU and GPU.
//! Transpose convolutions (also known as deconvolutions or fractionally-strided
//! convolutions) are commonly used in generator networks, autoencoders, and
//! upsampling layers for image super-resolution tasks.

use crate::tensor::TensorStorage;
use crate::{Result, Tensor, TensorError};
use num_traits::{One, Zero};

#[cfg(feature = "gpu")]
use wgpu::util::DeviceExt;

/// Performs 2D transposed convolution (deconvolution) operation
/// Input shape: [batch, in_channels, height, width] (NCHW format)
/// Weight shape: [in_channels, out_channels, kernel_height, kernel_width]
/// Output shape: [batch, out_channels, out_height, out_width]
pub fn conv_transpose2d<T>(
    input: &Tensor<T>,
    weight: &Tensor<T>,
    bias: Option<&Tensor<T>>,
    stride: (usize, usize),
    padding: (usize, usize),
    output_padding: (usize, usize),
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + One
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    match (&input.storage, &weight.storage) {
        (TensorStorage::Cpu(_input_arr), TensorStorage::Cpu(_weight_arr)) => {
            conv_transpose2d_cpu(input, weight, bias, stride, padding, output_padding)
        }
        #[cfg(feature = "gpu")]
        (TensorStorage::Gpu(_), TensorStorage::Gpu(_)) => {
            conv_transpose2d_gpu(input, weight, bias, stride, padding, output_padding)
        }
        #[cfg(feature = "gpu")]
        _ => Err(TensorError::unsupported_operation_simple(
            "Mixed CPU/GPU transposed convolution not supported".to_string(),
        )),
    }
}

// CPU implementation

fn conv_transpose2d_cpu<T>(
    input: &Tensor<T>,
    weight: &Tensor<T>,
    bias: Option<&Tensor<T>>,
    stride: (usize, usize),
    padding: (usize, usize),
    output_padding: (usize, usize),
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + One
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + Send
        + Sync
        + 'static,
{
    let input_shape = input.shape().dims();
    let weight_shape = weight.shape().dims();

    // Validate shapes
    if input_shape.len() != 4 {
        return Err(TensorError::invalid_shape_simple(
            "ConvTranspose2D input must be 4D (NCHW format)".to_string(),
        ));
    }
    if weight_shape.len() != 4 {
        return Err(TensorError::invalid_shape_simple(
            "ConvTranspose2D weight must be 4D".to_string(),
        ));
    }

    let batch_size = input_shape[0];
    let in_channels = input_shape[1];
    let input_height = input_shape[2];
    let input_width = input_shape[3];

    let weight_in_channels = weight_shape[0];
    let out_channels = weight_shape[1];
    let kernel_height = weight_shape[2];
    let kernel_width = weight_shape[3];

    if in_channels != weight_in_channels {
        return Err(TensorError::ShapeMismatch {
            operation: "conv_transpose2d".to_string(),
            expected: format!("weight in_channels={in_channels}"),
            got: format!("weight in_channels={weight_in_channels}"),
            context: None,
        });
    }

    // Calculate output dimensions
    let output_height =
        (input_height - 1) * stride.0 - 2 * padding.0 + kernel_height + output_padding.0;
    let output_width =
        (input_width - 1) * stride.1 - 2 * padding.1 + kernel_width + output_padding.1;

    // Get data arrays
    let input_data = input.as_slice().ok_or_else(|| {
        TensorError::device_error_simple("Cannot access input tensor data".to_string())
    })?;
    let weight_data = weight.as_slice().ok_or_else(|| {
        TensorError::device_error_simple("Cannot access weight tensor data".to_string())
    })?;

    let bias_data = if let Some(bias) = bias {
        Some(bias.as_slice().ok_or_else(|| {
            TensorError::device_error_simple("Cannot access bias tensor data".to_string())
        })?)
    } else {
        None
    };

    // Initialize output
    let output_size = batch_size * out_channels * output_height * output_width;
    let mut output_data = vec![T::zero(); output_size];

    // Perform transposed convolution
    for b in 0..batch_size {
        for oc in 0..out_channels {
            for ic in 0..in_channels {
                for iy in 0..input_height {
                    for ix in 0..input_width {
                        let input_idx =
                            ((b * in_channels + ic) * input_height + iy) * input_width + ix;
                        let input_val = input_data[input_idx].clone();

                        // Apply kernel to this input position
                        for kh in 0..kernel_height {
                            for kw in 0..kernel_width {
                                let out_y = iy * stride.0 + kh;
                                let out_x = ix * stride.1 + kw;

                                // Check bounds and padding
                                if out_y >= padding.0 && out_x >= padding.1 {
                                    let final_y = out_y - padding.0;
                                    let final_x = out_x - padding.1;

                                    if final_y < output_height && final_x < output_width {
                                        let weight_idx = ((ic * out_channels + oc) * kernel_height
                                            + kh)
                                            * kernel_width
                                            + kw;
                                        let output_idx = ((b * out_channels + oc) * output_height
                                            + final_y)
                                            * output_width
                                            + final_x;

                                        output_data[output_idx] = output_data[output_idx].clone()
                                            + input_val.clone() * weight_data[weight_idx].clone();
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Add bias
            if let Some(bias_data) = bias_data {
                for y in 0..output_height {
                    for x in 0..output_width {
                        let output_idx =
                            ((b * out_channels + oc) * output_height + y) * output_width + x;
                        output_data[output_idx] =
                            output_data[output_idx].clone() + bias_data[oc].clone();
                    }
                }
            }
        }
    }

    // Create output tensor using SciRS2's ndarray
    use scirs2_autograd::ndarray::Array;
    let output_array = Array::from_vec(output_data)
        .to_shape((batch_size, out_channels, output_height, output_width))
        .map_err(|e| TensorError::InvalidShape {
            operation: "conv_transpose2d".to_string(),
            reason: format!("Failed to reshape output: {e}"),
            shape: Some(vec![batch_size, out_channels, output_height, output_width]),
            context: None,
        })?
        .into_owned()
        .into_dyn();

    Ok(Tensor::from_storage(
        TensorStorage::Cpu(output_array),
        *input.device(),
    ))
}

// GPU implementation

#[cfg(feature = "gpu")]
fn conv_transpose2d_gpu<T>(
    input: &Tensor<T>,
    weight: &Tensor<T>,
    bias: Option<&Tensor<T>>,
    stride: (usize, usize),
    padding: (usize, usize),
    output_padding: (usize, usize),
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + One
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    use crate::gpu::buffer::GpuBuffer;

    let input_shape = input.shape().dims();
    let weight_shape = weight.shape().dims();

    let batch_size = input_shape[0];
    let in_channels = input_shape[1];
    let input_height = input_shape[2];
    let input_width = input_shape[3];

    let out_channels = weight_shape[1];
    let kernel_height = weight_shape[2];
    let kernel_width = weight_shape[3];

    // Calculate output dimensions
    let output_height =
        (input_height - 1) * stride.0 - 2 * padding.0 + kernel_height + output_padding.0;
    let output_width =
        (input_width - 1) * stride.1 - 2 * padding.1 + kernel_width + output_padding.1;

    if let (TensorStorage::Gpu(input_gpu), TensorStorage::Gpu(weight_gpu)) =
        (&input.storage, &weight.storage)
    {
        // Get GPU context
        let gpu_context = crate::device::get_gpu_context(0)?; // Use default GPU device 0
        let device = &gpu_context.device;
        let queue = &gpu_context.queue;

        // Create output buffer
        let output_size = batch_size * out_channels * output_height * output_width;
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("conv_transpose2d_output"),
            size: (output_size * std::mem::size_of::<T>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bias buffer
        let bias_buffer = if let Some(bias) = bias {
            if let TensorStorage::Gpu(bias_gpu) = &bias.storage {
                bias_gpu.buffer()
            } else {
                return Err(TensorError::unsupported_operation_simple(
                    "Mixed CPU/GPU bias not supported".to_string(),
                ));
            }
        } else {
            use wgpu::util::DeviceExt;
            let zeros = vec![T::zero(); out_channels];
            &(**device).create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("conv_transpose2d_zero_bias"),
                contents: bytemuck::cast_slice(&zeros),
                usage: wgpu::BufferUsages::STORAGE,
            })
        };

        // Create parameter buffer
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct ConvTransposeParams {
            batch_size: u32,
            in_channels: u32,
            input_height: u32,
            input_width: u32,
            out_channels: u32,
            kernel_height: u32,
            kernel_width: u32,
            output_height: u32,
            output_width: u32,
            stride_h: u32,
            stride_w: u32,
            pad_h: u32,
            pad_w: u32,
            output_pad_h: u32,
            output_pad_w: u32,
        }

        let params = ConvTransposeParams {
            batch_size: batch_size as u32,
            in_channels: in_channels as u32,
            input_height: input_height as u32,
            input_width: input_width as u32,
            out_channels: out_channels as u32,
            kernel_height: kernel_height as u32,
            kernel_width: kernel_width as u32,
            output_height: output_height as u32,
            output_width: output_width as u32,
            stride_h: stride.0 as u32,
            stride_w: stride.1 as u32,
            pad_h: padding.0 as u32,
            pad_w: padding.1 as u32,
            output_pad_h: output_padding.0 as u32,
            output_pad_w: output_padding.1 as u32,
        };

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("conv_transpose2d_params"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Load compute shader
        let shader_source = include_str!("../../gpu/shaders/conv_ops.wgsl");
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("conv_transpose2d_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("conv_transpose2d_bind_group_layout"),
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
            label: Some("conv_transpose2d_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_gpu.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weight_gpu.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bias_buffer.as_entire_binding(),
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
            label: Some("conv_transpose2d_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("conv_transpose2d_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("conv_transpose2d_kernel"),
            cache: None,
            compilation_options: Default::default(),
        });

        // Dispatch compute shader
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("conv_transpose2d_encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("conv_transpose2d_pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&compute_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch with workgroups for output spatial dimensions
            let workgroup_x = (output_width + 7) / 8;
            let workgroup_y = (output_height + 7) / 8;
            let workgroup_z = batch_size;

            compute_pass.dispatch_workgroups(
                workgroup_x as u32,
                workgroup_y as u32,
                workgroup_z as u32,
            );
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

        let result_gpu = crate::gpu::buffer::GpuBuffer::from_wgpu_buffer(
            output_buffer,
            device.clone(),
            queue.clone(),
            crate::Device::Gpu(device_id),
            output_size,
        );

        let mut result = Tensor::from_gpu_buffer(
            result_gpu,
            crate::Shape::new(vec![batch_size, out_channels, output_height, output_width]),
        );
        result.set_requires_grad(input.requires_grad());
        Ok(result)
    } else {
        Err(TensorError::unsupported_operation_simple(
            "GPU transposed convolution requires GPU tensors".to_string(),
        ))
    }
}
