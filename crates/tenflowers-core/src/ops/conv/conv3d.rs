//! 3D convolution operations
//!
//! This module provides 3D convolution implementations for both CPU and GPU.
//! 3D convolutions are commonly used for video processing, volumetric data,
//! and 3D medical imaging applications.

use crate::tensor::TensorStorage;
use crate::{Result, Tensor, TensorError};
use num_traits::{One, Zero};
use scirs2_autograd::ndarray::{ArrayD, IxDyn};

#[cfg(feature = "gpu")]
use wgpu::util::DeviceExt;

/// Performs 3D convolution operation
/// Input shape: [batch, in_channels, depth, height, width] (NCDHW format)
/// Weight shape: [out_channels, in_channels, kernel_depth, kernel_height, kernel_width]
/// Output shape: [batch, out_channels, out_depth, out_height, out_width]
pub fn conv3d<T>(
    input: &Tensor<T>,
    weight: &Tensor<T>,
    bias: Option<&Tensor<T>>,
    stride: (usize, usize, usize),
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
            conv3d_cpu(input, weight, bias, stride, padding)
        }
        #[cfg(feature = "gpu")]
        (TensorStorage::Gpu(input_gpu), TensorStorage::Gpu(weight_gpu)) => {
            conv3d_gpu(input, weight, bias, stride, padding)
        }
        #[cfg(feature = "gpu")]
        _ => Err(TensorError::unsupported_operation_simple(
            "Mixed CPU/GPU convolution not supported".to_string(),
        )),
    }
}

// CPU implementation

fn conv3d_cpu<T>(
    input: &Tensor<T>,
    weight: &Tensor<T>,
    bias: Option<&Tensor<T>>,
    stride: (usize, usize, usize),
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
                "GPU conv3d not yet implemented".to_string(),
            ));
        }
    };
    #[allow(clippy::infallible_destructuring_match)]
    let weight_arr = match &weight.storage {
        TensorStorage::Cpu(arr) => arr,
        #[cfg(feature = "gpu")]
        TensorStorage::Gpu(_) => {
            return Err(TensorError::unsupported_operation_simple(
                "GPU conv3d not yet implemented".to_string(),
            ));
        }
    };

    // Validate input shapes
    if input_arr.ndim() != 5 {
        return Err(TensorError::invalid_shape_simple(
            "Conv3D input must be 5D (NCDHW format)".to_string(),
        ));
    }
    if weight_arr.ndim() != 5 {
        return Err(TensorError::invalid_shape_simple(
            "Conv3D weight must be 5D".to_string(),
        ));
    }

    let input_shape = input_arr.shape();
    let weight_shape = weight_arr.shape();

    let batch_size = input_shape[0];
    let in_channels = input_shape[1];
    let in_depth = input_shape[2];
    let in_height = input_shape[3];
    let in_width = input_shape[4];

    let out_channels = weight_shape[0];
    let weight_in_channels = weight_shape[1];
    let kernel_depth = weight_shape[2];
    let kernel_height = weight_shape[3];
    let kernel_width = weight_shape[4];

    if in_channels != weight_in_channels {
        return Err(TensorError::ShapeMismatch {
            operation: "conv3d".to_string(),
            expected: format!("weight in_channels={in_channels}"),
            got: format!("weight in_channels={weight_in_channels}"),
            context: None,
        });
    }

    // Calculate output dimensions based on padding
    let (out_depth, out_height, out_width, pad_front, pad_top, pad_left) = match padding {
        "valid" => {
            let out_d = (in_depth - kernel_depth) / stride.0 + 1;
            let out_h = (in_height - kernel_height) / stride.1 + 1;
            let out_w = (in_width - kernel_width) / stride.2 + 1;
            (out_d, out_h, out_w, 0, 0, 0)
        }
        "same" => {
            let out_d = (in_depth + stride.0 - 1) / stride.0;
            let out_h = (in_height + stride.1 - 1) / stride.1;
            let out_w = (in_width + stride.2 - 1) / stride.2;
            let pad_d = std::cmp::max(0, (out_d - 1) * stride.0 + kernel_depth - in_depth);
            let pad_h = std::cmp::max(0, (out_h - 1) * stride.1 + kernel_height - in_height);
            let pad_w = std::cmp::max(0, (out_w - 1) * stride.2 + kernel_width - in_width);
            (out_d, out_h, out_w, pad_d / 2, pad_h / 2, pad_w / 2)
        }
        _ => {
            return Err(TensorError::invalid_argument(format!(
                "Unknown padding mode: {padding}"
            )))
        }
    };

    // Create output tensor
    let mut output = ArrayD::<T>::zeros(IxDyn(&[
        batch_size,
        out_channels,
        out_depth,
        out_height,
        out_width,
    ]));

    // Perform convolution
    for b in 0..batch_size {
        for oc in 0..out_channels {
            for od in 0..out_depth {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        let mut sum = T::zero();

                        for ic in 0..in_channels {
                            for kd in 0..kernel_depth {
                                for kh in 0..kernel_height {
                                    for kw in 0..kernel_width {
                                        let id = od * stride.0 + kd;
                                        let ih = oh * stride.1 + kh;
                                        let iw = ow * stride.2 + kw;

                                        // Apply padding
                                        if id >= pad_front
                                            && id < in_depth + pad_front
                                            && ih >= pad_top
                                            && ih < in_height + pad_top
                                            && iw >= pad_left
                                            && iw < in_width + pad_left
                                        {
                                            let id_actual = id - pad_front;
                                            let ih_actual = ih - pad_top;
                                            let iw_actual = iw - pad_left;

                                            if id_actual < in_depth
                                                && ih_actual < in_height
                                                && iw_actual < in_width
                                            {
                                                let input_val = input_arr
                                                    [[b, ic, id_actual, ih_actual, iw_actual]];
                                                let weight_val = weight_arr[[oc, ic, kd, kh, kw]];
                                                sum = sum + (input_val * weight_val);
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        output[[b, oc, od, oh, ow]] = sum;
                    }
                }
            }
        }
    }

    // Add bias if provided
    if let Some(bias) = bias {
        match &bias.storage {
            TensorStorage::Cpu(bias_arr) => {
                if bias_arr.ndim() != 1 || bias_arr.shape()[0] != out_channels {
                    return Err(TensorError::ShapeMismatch {
                        operation: "conv3d".to_string(),
                        expected: format!("bias shape [{out_channels}]"),
                        got: format!("bias shape {:?}", bias_arr.shape()),
                        context: None,
                    });
                }

                // Add bias to each output channel
                for b in 0..batch_size {
                    for oc in 0..out_channels {
                        let bias_val = bias_arr[[oc]];
                        for od in 0..out_depth {
                            for oh in 0..out_height {
                                for ow in 0..out_width {
                                    output[[b, oc, od, oh, ow]] =
                                        output[[b, oc, od, oh, ow]] + bias_val;
                                }
                            }
                        }
                    }
                }
            }
            #[cfg(feature = "gpu")]
            TensorStorage::Gpu(bias_gpu) => {
                // Convert GPU bias to CPU for this CPU convolution
                let bias_cpu = bias_gpu.to_cpu()?;
                let bias_arr = scirs2_autograd::ndarray::Array1::from_vec(bias_cpu).into_dyn();

                if bias_arr.ndim() != 1 || bias_arr.shape()[0] != out_channels {
                    return Err(TensorError::ShapeMismatch {
                        operation: "conv3d".to_string(),
                        expected: format!("bias shape [{out_channels}]"),
                        got: format!("bias shape {:?}", bias_arr.shape()),
                        context: None,
                    });
                }

                // Add bias to each output channel
                for b in 0..batch_size {
                    for oc in 0..out_channels {
                        let bias_val = bias_arr[[oc]];
                        for od in 0..out_depth {
                            for oh in 0..out_height {
                                for ow in 0..out_width {
                                    output[[b, oc, od, oh, ow]] =
                                        output[[b, oc, od, oh, ow]] + bias_val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(Tensor::from_array(output))
}

// GPU implementation

#[cfg(feature = "gpu")]
fn conv3d_gpu<T>(
    input: &Tensor<T>,
    weight: &Tensor<T>,
    bias: Option<&Tensor<T>>,
    stride: (usize, usize, usize),
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

    let input_shape = input.shape();
    let weight_shape = weight.shape();

    let batch_size = input_shape.dims()[0];
    let in_channels = input_shape.dims()[1];
    let in_depth = input_shape.dims()[2];
    let in_height = input_shape.dims()[3];
    let in_width = input_shape.dims()[4];

    let out_channels = weight_shape.dims()[0];
    let kernel_depth = weight_shape.dims()[2];
    let kernel_height = weight_shape.dims()[3];
    let kernel_width = weight_shape.dims()[4];

    // Calculate output dimensions
    let (out_depth, out_height, out_width, pad_front, pad_top, pad_left) = match padding {
        "valid" => {
            let out_d = (in_depth - kernel_depth) / stride.0 + 1;
            let out_h = (in_height - kernel_height) / stride.1 + 1;
            let out_w = (in_width - kernel_width) / stride.2 + 1;
            (out_d, out_h, out_w, 0, 0, 0)
        }
        "same" => {
            let out_d = (in_depth + stride.0 - 1) / stride.0;
            let out_h = (in_height + stride.1 - 1) / stride.1;
            let out_w = (in_width + stride.2 - 1) / stride.2;
            let pad_d = std::cmp::max(0, (out_d - 1) * stride.0 + kernel_depth - in_depth);
            let pad_h = std::cmp::max(0, (out_h - 1) * stride.1 + kernel_height - in_height);
            let pad_w = std::cmp::max(0, (out_w - 1) * stride.2 + kernel_width - in_width);
            (out_d, out_h, out_w, pad_d / 2, pad_h / 2, pad_w / 2)
        }
        _ => {
            return Err(TensorError::invalid_argument(format!(
                "Unknown padding mode: {padding}"
            )))
        }
    };

    if let (TensorStorage::Gpu(input_gpu), TensorStorage::Gpu(weight_gpu)) =
        (&input.storage, &weight.storage)
    {
        // Get GPU context
        let gpu_context = crate::device::get_gpu_context(0)?; // Use default GPU device 0
        let device = &gpu_context.device;
        let queue = &gpu_context.queue;

        // Create output buffer
        let output_size = batch_size * out_channels * out_depth * out_height * out_width;
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("conv3d_output"),
            size: (output_size * std::mem::size_of::<T>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bias buffer (use zeros if no bias provided)
        let bias_data = if let Some(bias) = bias {
            if let TensorStorage::Gpu(bias_gpu) = &bias.storage {
                bias_gpu.buffer()
            } else {
                return Err(TensorError::unsupported_operation_simple(
                    "Mixed CPU/GPU bias not supported".to_string(),
                ));
            }
        } else {
            // Create zero bias buffer
            use wgpu::util::DeviceExt;
            let zeros = vec![T::zero(); out_channels];
            &(**device).create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("conv3d_zero_bias"),
                contents: bytemuck::cast_slice(&zeros),
                usage: wgpu::BufferUsages::STORAGE,
            })
        };

        // Create parameter buffer
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct Conv3dParams {
            batch_size: u32,
            in_channels: u32,
            input_depth: u32,
            input_height: u32,
            input_width: u32,
            out_channels: u32,
            kernel_depth: u32,
            kernel_height: u32,
            kernel_width: u32,
            output_depth: u32,
            output_height: u32,
            output_width: u32,
            stride_d: u32,
            stride_h: u32,
            stride_w: u32,
            pad_d: u32,
            pad_h: u32,
            pad_w: u32,
        }

        let params = Conv3dParams {
            batch_size: batch_size as u32,
            in_channels: in_channels as u32,
            input_depth: in_depth as u32,
            input_height: in_height as u32,
            input_width: in_width as u32,
            out_channels: out_channels as u32,
            kernel_depth: kernel_depth as u32,
            kernel_height: kernel_height as u32,
            kernel_width: kernel_width as u32,
            output_depth: out_depth as u32,
            output_height: out_height as u32,
            output_width: out_width as u32,
            stride_d: stride.0 as u32,
            stride_h: stride.1 as u32,
            stride_w: stride.2 as u32,
            pad_d: pad_front as u32,
            pad_h: pad_top as u32,
            pad_w: pad_left as u32,
        };

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("conv3d_params"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Load compute shader
        let shader_source = include_str!("../../gpu/shaders/conv_ops.wgsl");
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("conv3d_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("conv3d_bind_group_layout"),
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
            label: Some("conv3d_bind_group"),
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
                    resource: bias_data.as_entire_binding(),
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
            label: Some("conv3d_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("conv3d_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("conv3d_kernel"),
            cache: None,
            compilation_options: Default::default(),
        });

        // Dispatch compute shader
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("conv3d_encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("conv3d_pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&compute_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch with 3D workgroups for output spatial dimensions
            let workgroup_x = (out_width + 7) / 8;
            let workgroup_y = (out_height + 7) / 8;
            let workgroup_z = (out_depth + 3) / 4;
            let num_batches = batch_size * out_channels;

            // Need to dispatch multiple times if batch_size * out_channels > max workgroup z
            let max_z = 65535; // WebGPU limit
            let batches_per_dispatch = std::cmp::min(num_batches, max_z);

            for batch_offset in (0..num_batches).step_by(batches_per_dispatch) {
                let current_batches =
                    std::cmp::min(batches_per_dispatch, num_batches - batch_offset);
                compute_pass.dispatch_workgroups(
                    workgroup_x as u32,
                    workgroup_y as u32,
                    (workgroup_z * current_batches) as u32,
                );
            }
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
            crate::Shape::new(vec![
                batch_size,
                out_channels,
                out_depth,
                out_height,
                out_width,
            ]),
        );
        result.set_requires_grad(input.requires_grad());
        Ok(result)
    } else {
        Err(TensorError::unsupported_operation_simple(
            "GPU convolution requires GPU tensors".to_string(),
        ))
    }
}
