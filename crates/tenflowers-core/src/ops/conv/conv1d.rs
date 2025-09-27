//! 1D convolution operations
//!
//! This module provides 1D convolution implementations for both CPU and GPU.
//! Supports standard convolution with optional bias and configurable padding.

#![allow(clippy::clone_on_copy)]

use crate::tensor::TensorStorage;
use crate::{Result, Tensor, TensorError};
use num_traits::{One, Zero};
use scirs2_autograd::ndarray::{ArrayD, IxDyn};

/// Performs 1D convolution operation
/// Input shape: [batch, in_channels, length] (NCL format)
/// Weight shape: [out_channels, in_channels, kernel_length]
/// Output shape: [batch, out_channels, out_length]
pub fn conv1d<T>(
    input: &Tensor<T>,
    weight: &Tensor<T>,
    bias: Option<&Tensor<T>>,
    stride: usize,
    padding: &str,
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
            conv1d_cpu(input, weight, bias, stride, padding)
        }
        #[cfg(feature = "gpu")]
        (TensorStorage::Gpu(_input_gpu), TensorStorage::Gpu(_weight_gpu)) => {
            conv1d_gpu(input, weight, bias, stride, padding)
        }
        #[allow(unreachable_patterns)]
        _ => Err(TensorError::invalid_argument(
            "Mixed CPU/GPU conv1d not supported".to_string(),
        )),
    }
}

/// CPU implementation of 1D convolution
fn conv1d_cpu<T>(
    input: &Tensor<T>,
    weight: &Tensor<T>,
    bias: Option<&Tensor<T>>,
    stride: usize,
    padding: &str,
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
    #[allow(clippy::infallible_destructuring_match)]
    let input_arr = match &input.storage {
        TensorStorage::Cpu(arr) => arr,
        #[cfg(feature = "gpu")]
        TensorStorage::Gpu(_) => {
            return Err(TensorError::unsupported_operation_simple(
                "GPU conv1d not yet implemented".to_string(),
            ));
        }
    };
    #[allow(clippy::infallible_destructuring_match)]
    let weight_arr = match &weight.storage {
        TensorStorage::Cpu(arr) => arr,
        #[cfg(feature = "gpu")]
        TensorStorage::Gpu(_) => {
            return Err(TensorError::unsupported_operation_simple(
                "GPU conv1d not yet implemented".to_string(),
            ));
        }
    };

    // Validate input shapes
    if input_arr.ndim() != 3 {
        return Err(TensorError::invalid_shape_simple(
            "Conv1D input must be 3D (NCL format)".to_string(),
        ));
    }
    if weight_arr.ndim() != 3 {
        return Err(TensorError::invalid_shape_simple(
            "Conv1D weight must be 3D".to_string(),
        ));
    }

    let input_shape = input_arr.shape();
    let weight_shape = weight_arr.shape();

    let batch_size = input_shape[0];
    let in_channels = input_shape[1];
    let in_length = input_shape[2];

    let out_channels = weight_shape[0];
    let weight_in_channels = weight_shape[1];
    let kernel_length = weight_shape[2];

    if in_channels != weight_in_channels {
        return Err(TensorError::shape_mismatch(
            "conv",
            &format!("weight in_channels={in_channels}"),
            &format!("weight in_channels={weight_in_channels}"),
        ));
    }

    // Calculate output dimensions based on padding
    let (out_length, pad_left) = match padding {
        "valid" => {
            let out_len = (in_length - kernel_length) / stride + 1;
            (out_len, 0)
        }
        "same" => {
            let out_len = (in_length + stride - 1) / stride;
            let pad_total = std::cmp::max(0, (out_len - 1) * stride + kernel_length - in_length);
            (out_len, pad_total / 2)
        }
        _ => {
            return Err(TensorError::invalid_argument(format!(
                "Unknown padding mode: {padding}"
            )))
        }
    };

    // Create output tensor
    let mut output = ArrayD::<T>::zeros(IxDyn(&[batch_size, out_channels, out_length]));

    // Perform convolution
    for b in 0..batch_size {
        for oc in 0..out_channels {
            for ol in 0..out_length {
                let mut sum = T::zero();

                for ic in 0..in_channels {
                    for k in 0..kernel_length {
                        let il = ol * stride + k;

                        // Apply padding
                        if il >= pad_left && il < in_length + pad_left {
                            let il_actual = il - pad_left;

                            if il_actual < in_length {
                                let input_val = input_arr[[b, ic, il_actual]].clone();
                                let weight_val = weight_arr[[oc, ic, k]].clone();
                                sum = sum + (input_val * weight_val);
                            }
                        }
                    }
                }

                output[[b, oc, ol]] = sum;
            }
        }
    }

    // Apply bias if provided
    if let Some(bias) = bias {
        match &bias.storage {
            TensorStorage::Cpu(bias_arr) => {
                if bias_arr.shape() != [out_channels] {
                    return Err(TensorError::shape_mismatch(
                        "conv",
                        &format!("bias shape [{out_channels}]"),
                        &format!("bias shape {:?}", bias_arr.shape()),
                    ));
                }

                // Add bias to each output channel
                for b in 0..batch_size {
                    for oc in 0..out_channels {
                        let bias_val = bias_arr[[oc]].clone();
                        for ol in 0..out_length {
                            output[[b, oc, ol]] = output[[b, oc, ol]].clone() + bias_val.clone();
                        }
                    }
                }
            }
            #[cfg(feature = "gpu")]
            TensorStorage::Gpu(bias_gpu) => {
                // Convert GPU bias to CPU for this CPU convolution
                let bias_cpu = bias_gpu.to_cpu()?;
                let bias_arr = scirs2_autograd::ndarray::Array1::from_vec(bias_cpu).into_dyn();

                if bias_arr.shape() != [out_channels] {
                    return Err(TensorError::shape_mismatch(
                        "conv",
                        &format!("bias shape [{out_channels}]"),
                        &format!("bias shape {:?}", bias_arr.shape()),
                    ));
                }

                // Add bias to each output channel
                for b in 0..batch_size {
                    for oc in 0..out_channels {
                        let bias_val = bias_arr[[oc]].clone();
                        for ol in 0..out_length {
                            output[[b, oc, ol]] = output[[b, oc, ol]].clone() + bias_val.clone();
                        }
                    }
                }
            }
        }
    }

    Ok(Tensor::from_array(output))
}

#[cfg(feature = "gpu")]
/// GPU implementation of 1D convolution
fn conv1d_gpu<T>(
    input: &Tensor<T>,
    weight: &Tensor<T>,
    bias: Option<&Tensor<T>>,
    stride: usize,
    padding: &str,
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
    use crate::tensor::TensorStorage;

    let input_shape = input.shape();
    let weight_shape = weight.shape();

    let batch_size = input_shape.dims()[0];
    let in_channels = input_shape.dims()[1];
    let in_length = input_shape.dims()[2];

    let out_channels = weight_shape.dims()[0];
    let weight_in_channels = weight_shape.dims()[1];
    let kernel_length = weight_shape.dims()[2];

    if in_channels != weight_in_channels {
        return Err(TensorError::shape_mismatch(
            "conv",
            &format!("weight in_channels={in_channels}"),
            &format!("weight in_channels={weight_in_channels}"),
        ));
    }

    // Calculate output dimensions based on padding
    let (out_length, pad_left) = match padding {
        "valid" => {
            let out_len = (in_length - kernel_length) / stride + 1;
            (out_len, 0)
        }
        "same" => {
            let out_len = (in_length + stride - 1) / stride;
            let pad_total = std::cmp::max(0, (out_len - 1) * stride + kernel_length - in_length);
            (out_len, pad_total / 2)
        }
        _ => {
            return Err(TensorError::invalid_argument(format!(
                "Unknown padding mode: {padding}"
            )))
        }
    };

    // Get GPU context
    let gpu_context = crate::device::get_gpu_context(0)?; // Use default GPU device 0
    let device = &gpu_context.device;
    let queue = &gpu_context.queue;

    // Get GPU buffers for input and weight
    let input_gpu = match &input.storage {
        TensorStorage::Gpu(buf) => buf,
        _ => {
            return Err(TensorError::invalid_argument(
                "GPU conv1d requires GPU input tensor".to_string(),
            ))
        }
    };

    let weight_gpu = match &weight.storage {
        TensorStorage::Gpu(buf) => buf,
        _ => {
            return Err(TensorError::invalid_argument(
                "GPU conv1d requires GPU weight tensor".to_string(),
            ))
        }
    };

    // Handle bias
    let bias_data = if let Some(bias) = bias {
        match &bias.storage {
            TensorStorage::Gpu(bias_gpu) => bias_gpu.to_cpu()?.to_vec(),
            TensorStorage::Cpu(bias_arr) => bias_arr.iter().cloned().collect(),
        }
    } else {
        vec![T::zero(); out_channels]
    };

    // Create output buffer
    let output_size = batch_size * out_channels * out_length;
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("conv1d_output"),
        size: (output_size * std::mem::size_of::<T>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Execute Conv1D GPU kernel
    execute_conv1d_kernel(
        device,
        queue,
        input_gpu,
        weight_gpu,
        &bias_data,
        &output_buffer,
        batch_size,
        in_channels,
        out_channels,
        in_length,
        out_length,
        kernel_length,
        stride,
        pad_left,
    )?;

    // Read back result and create output tensor
    let output_gpu = crate::gpu::buffer::GpuBuffer::from_wgpu_buffer(
        output_buffer,
        device.clone(),
        queue.clone(),
        crate::Device::Gpu(0),
        output_size,
    );
    Ok(Tensor::from_gpu_buffer(
        output_gpu,
        crate::Shape::new(vec![batch_size, out_channels, out_length]),
    ))
}

#[cfg(feature = "gpu")]
/// Execute Conv1D GPU kernel
fn execute_conv1d_kernel<T>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    input_gpu: &crate::gpu::buffer::GpuBuffer<T>,
    weight_gpu: &crate::gpu::buffer::GpuBuffer<T>,
    bias_data: &[T],
    output_buffer: &wgpu::Buffer,
    batch_size: usize,
    in_channels: usize,
    out_channels: usize,
    in_length: usize,
    out_length: usize,
    kernel_length: usize,
    stride: usize,
    pad_left: usize,
) -> Result<()>
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
    use wgpu::util::DeviceExt;

    // Reuse ConvParams structure for compatibility with shader
    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct ConvParams {
        batch_size: u32,
        in_channels: u32,
        input_height: u32, // Reused for in_length
        input_width: u32,  // Unused for 1D, set to 1
        out_channels: u32,
        kernel_height: u32, // Reused for kernel_length
        kernel_width: u32,  // Unused for 1D, set to 1
        output_height: u32, // Reused for out_length
        output_width: u32,  // Unused for 1D, set to 1
        stride_h: u32,      // Reused for stride
        stride_w: u32,      // Unused for 1D, set to 1
        pad_h: u32,         // Reused for pad_left
        pad_w: u32,         // Unused for 1D, set to 0
    }

    let conv1d_params = ConvParams {
        batch_size: batch_size as u32,
        in_channels: in_channels as u32,
        input_height: in_length as u32, // Map in_length to input_height
        input_width: 1,                 // Unused
        out_channels: out_channels as u32,
        kernel_height: kernel_length as u32, // Map kernel_length to kernel_height
        kernel_width: 1,                     // Unused
        output_height: out_length as u32,    // Map out_length to output_height
        output_width: 1,                     // Unused
        stride_h: stride as u32,             // Map stride to stride_h
        stride_w: 1,                         // Unused
        pad_h: pad_left as u32,              // Map pad_left to pad_h
        pad_w: 0,                            // Unused
    };

    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("conv1d_params"),
        contents: bytemuck::cast_slice(&[conv1d_params]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bias_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("conv1d_bias"),
        contents: bytemuck::cast_slice(bias_data),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // Load and create compute shader
    let shader_source = include_str!("../../gpu/shaders/conv_ops.wgsl");
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("conv1d_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    // Create bind group layout
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("conv1d_bind_group_layout"),
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

    // Create compute pipeline
    let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("conv1d_pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("conv1d_pipeline"),
        layout: Some(&compute_pipeline_layout),
        module: &shader,
        entry_point: Some("conv1d_kernel"),
        cache: None,
        compilation_options: Default::default(),
    });

    // Create bind group
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("conv1d_bind_group"),
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

    // Dispatch compute shader
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("conv1d_encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("conv1d_pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        // Calculate dispatch size
        let workgroup_size_x = 8u32;
        let workgroup_size_y = 8u32;
        let dispatch_x = (out_length as u32 + workgroup_size_x - 1) / workgroup_size_x;
        let dispatch_y = (out_channels as u32 + workgroup_size_y - 1) / workgroup_size_y;
        let dispatch_z = batch_size as u32;

        compute_pass.dispatch_workgroups(dispatch_x, dispatch_y, dispatch_z);
    }

    queue.submit(std::iter::once(encoder.finish()));

    Ok(())
}
