//! 2D convolution operations
//!
//! This module provides 2D convolution implementations for both CPU and GPU.
//! GPU implementation includes multiple optimization strategies: standard,
//! Winograd, FFT-based, tiled, and im2col approaches for different input sizes.

#![allow(clippy::clone_on_copy)]

use crate::layout::{convert_layout, DataLayout, LayoutOptimizer, OperationType};
use crate::tensor::TensorStorage;
use crate::{Result, Tensor, TensorError};
use num_traits::{One, Zero};
use scirs2_autograd::ndarray::{ArrayD, IxDyn};

#[cfg(feature = "gpu")]
use wgpu::util::DeviceExt;

/// Performs 2D convolution operation
/// Input shape: [batch, in_channels, height, width] (NCHW format)
/// Weight shape: [out_channels, in_channels, kernel_height, kernel_width]
/// Output shape: [batch, out_channels, out_height, out_width]
pub fn conv2d<T>(
    input: &Tensor<T>,
    weight: &Tensor<T>,
    bias: Option<&Tensor<T>>,
    stride: (usize, usize),
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
            conv2d_cpu(input, weight, bias, stride, padding)
        }
        #[cfg(feature = "gpu")]
        (TensorStorage::Gpu(input_gpu), TensorStorage::Gpu(weight_gpu)) => {
            conv2d_gpu(input, weight, bias, stride, padding)
        }
        #[cfg(feature = "gpu")]
        _ => Err(TensorError::unsupported_operation_simple(
            "Mixed CPU/GPU convolution not supported".to_string(),
        )),
    }
}

/// 2D convolution with explicit layout control
/// Allows specifying input and output data layouts for optimization
pub fn conv2d_with_layout<T>(
    input: &Tensor<T>,
    weight: &Tensor<T>,
    bias: Option<&Tensor<T>>,
    stride: (usize, usize),
    padding: &str,
    input_layout: DataLayout,
    output_layout: DataLayout,
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
    // Convert input to optimal layout if needed (assuming current layout is NCHW)
    let current_layout = DataLayout::NCHW;
    let optimized_input = if input_layout != current_layout {
        convert_layout(input, current_layout, input_layout)?
    } else {
        input.clone()
    };

    // Perform convolution
    let result = conv2d(&optimized_input, weight, bias, stride, padding)?;

    // Convert output to requested layout if needed
    if output_layout != input_layout {
        convert_layout(&result, input_layout, output_layout)
    } else {
        Ok(result)
    }
}

/// Auto-layout 2D convolution that automatically selects optimal layouts
/// Analyzes input characteristics and selects best layout strategy
pub fn conv2d_auto_layout<T>(
    input: &Tensor<T>,
    weight: &Tensor<T>,
    bias: Option<&Tensor<T>>,
    stride: (usize, usize),
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
    let optimizer = LayoutOptimizer::default();
    let optimal_layout = optimizer.preferred_layout(input.device(), OperationType::Convolution);

    conv2d_with_layout(
        input,
        weight,
        bias,
        stride,
        padding,
        optimal_layout,
        optimal_layout,
    )
}

// CPU implementation

fn conv2d_cpu<T>(
    input: &Tensor<T>,
    weight: &Tensor<T>,
    bias: Option<&Tensor<T>>,
    stride: (usize, usize),
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
                "GPU conv2d not yet implemented".to_string(),
            ));
        }
    };
    #[allow(clippy::infallible_destructuring_match)]
    let weight_arr = match &weight.storage {
        TensorStorage::Cpu(arr) => arr,
        #[cfg(feature = "gpu")]
        TensorStorage::Gpu(_) => {
            return Err(TensorError::unsupported_operation_simple(
                "GPU conv2d not yet implemented".to_string(),
            ));
        }
    };

    // Validate input shapes
    if input_arr.ndim() != 4 {
        return Err(TensorError::invalid_shape_simple(
            "Conv2D input must be 4D (NCHW format)".to_string(),
        ));
    }
    if weight_arr.ndim() != 4 {
        return Err(TensorError::invalid_shape_simple(
            "Conv2D weight must be 4D".to_string(),
        ));
    }

    let input_shape = input_arr.shape();
    let weight_shape = weight_arr.shape();

    let batch_size = input_shape[0];
    let in_channels = input_shape[1];
    let in_height = input_shape[2];
    let in_width = input_shape[3];

    let out_channels = weight_shape[0];
    let weight_in_channels = weight_shape[1];
    let kernel_height = weight_shape[2];
    let kernel_width = weight_shape[3];

    if in_channels != weight_in_channels {
        return Err(TensorError::ShapeMismatch {
            operation: "conv2d".to_string(),
            expected: format!("weight in_channels={in_channels}"),
            got: format!("weight in_channels={weight_in_channels}"),
            context: None,
        });
    }

    // Calculate output dimensions based on padding
    let (out_height, out_width, pad_top, pad_left) = match padding {
        "valid" => {
            let out_h = (in_height - kernel_height) / stride.0 + 1;
            let out_w = (in_width - kernel_width) / stride.1 + 1;
            (out_h, out_w, 0, 0)
        }
        "same" => {
            let out_h = (in_height + stride.0 - 1) / stride.0;
            let out_w = (in_width + stride.1 - 1) / stride.1;
            let pad_h = std::cmp::max(0, (out_h - 1) * stride.0 + kernel_height - in_height);
            let pad_w = std::cmp::max(0, (out_w - 1) * stride.1 + kernel_width - in_width);
            (out_h, out_w, pad_h / 2, pad_w / 2)
        }
        _ => {
            return Err(TensorError::invalid_argument(format!(
                "Unknown padding mode: {padding}"
            )))
        }
    };

    // Create output tensor
    let mut output = ArrayD::<T>::zeros(IxDyn(&[batch_size, out_channels, out_height, out_width]));

    // Perform convolution
    for b in 0..batch_size {
        for oc in 0..out_channels {
            for oh in 0..out_height {
                for ow in 0..out_width {
                    let mut sum = T::zero();

                    for ic in 0..in_channels {
                        for kh in 0..kernel_height {
                            for kw in 0..kernel_width {
                                let ih = oh * stride.0 + kh;
                                let iw = ow * stride.1 + kw;

                                // Apply padding
                                if ih >= pad_top
                                    && ih < in_height + pad_top
                                    && iw >= pad_left
                                    && iw < in_width + pad_left
                                {
                                    let ih_actual = ih - pad_top;
                                    let iw_actual = iw - pad_left;

                                    if ih_actual < in_height && iw_actual < in_width {
                                        let input_val =
                                            input_arr[[b, ic, ih_actual, iw_actual]].clone();
                                        let weight_val = weight_arr[[oc, ic, kh, kw]].clone();
                                        sum = sum + (input_val * weight_val);
                                    }
                                }
                            }
                        }
                    }

                    output[[b, oc, oh, ow]] = sum;
                }
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
                        for oh in 0..out_height {
                            for ow in 0..out_width {
                                output[[b, oc, oh, ow]] =
                                    output[[b, oc, oh, ow]].clone() + bias_val.clone();
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

                if bias_arr.shape() != [out_channels] {
                    return Err(TensorError::ShapeMismatch {
                        operation: "conv2d".to_string(),
                        expected: format!("bias shape [{out_channels}]"),
                        got: format!("bias shape {:?}", bias_arr.shape()),
                        context: None,
                    });
                }

                // Add bias to each output channel
                for b in 0..batch_size {
                    for oc in 0..out_channels {
                        let bias_val = bias_arr[[oc]].clone();
                        for oh in 0..out_height {
                            for ow in 0..out_width {
                                output[[b, oc, oh, ow]] =
                                    output[[b, oc, oh, ow]].clone() + bias_val.clone();
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(Tensor::from_array(output))
}

// GPU implementation with optimization strategies

#[cfg(feature = "gpu")]
fn execute_im2col_convolution<T>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    input_gpu: &crate::gpu::buffer::GpuBuffer<T>,
    weight_gpu: &crate::gpu::buffer::GpuBuffer<T>,
    bias_data: &wgpu::Buffer,
    output_buffer: &wgpu::Buffer,
    kernel_type: ConvKernelType,
    batch_size: usize,
    in_channels: usize,
    out_channels: usize,
    in_height: usize,
    in_width: usize,
    out_height: usize,
    out_width: usize,
    kernel_height: usize,
    kernel_width: usize,
    stride: (usize, usize),
    pad_top: usize,
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

    // Create Im2Col parameters
    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct Im2ColParams {
        batch_size: u32,
        in_channels: u32,
        out_channels: u32,
        input_height: u32,
        input_width: u32,
        output_height: u32,
        output_width: u32,
        kernel_height: u32,
        kernel_width: u32,
        stride_h: u32,
        stride_w: u32,
        pad_h: u32,
        pad_w: u32,
        dilation_h: u32,
        dilation_w: u32,
    }

    let im2col_params = Im2ColParams {
        batch_size: batch_size as u32,
        in_channels: in_channels as u32,
        out_channels: out_channels as u32,
        input_height: in_height as u32,
        input_width: in_width as u32,
        output_height: out_height as u32,
        output_width: out_width as u32,
        kernel_height: kernel_height as u32,
        kernel_width: kernel_width as u32,
        stride_h: stride.0 as u32,
        stride_w: stride.1 as u32,
        pad_h: pad_top as u32,
        pad_w: pad_left as u32,
        dilation_h: 1,
        dilation_w: 1,
    };

    let im2col_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("im2col_params"),
        contents: bytemuck::cast_slice(&[im2col_params]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // Create intermediate im2col matrix buffer
    let kernel_size = kernel_height * kernel_width;
    let output_size = out_height * out_width;
    let matrix_height = in_channels * kernel_size;
    let matrix_size = batch_size * matrix_height * output_size;

    let im2col_matrix_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("im2col_matrix"),
        size: (matrix_size * std::mem::size_of::<T>()) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Load compute shader
    let shader_source = include_str!("../../gpu/shaders/conv_ops.wgsl");
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("im2col_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    // Create bind group layout for Im2Col
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("im2col_bind_group_layout"),
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
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 5,
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
        label: Some("im2col_bind_group"),
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
                resource: im2col_matrix_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: im2col_params_buffer.as_entire_binding(),
            },
        ],
    });

    // Create compute pipeline
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("im2col_pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("im2col_encoder"),
    });

    // Step 1: Im2Col transformation
    {
        let transform_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("im2col_transform_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("im2col_coalesced_transform"),
            cache: None,
            compilation_options: Default::default(),
        });

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("im2col_transform_pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&transform_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        // Dispatch for im2col transformation
        let total_elements = kernel_size * output_size;
        let workgroup_x = (total_elements + 255) / 256;
        let workgroup_z = batch_size * in_channels;
        compute_pass.dispatch_workgroups(workgroup_x as u32, 1, workgroup_z as u32);
    }

    // Step 2: Matrix multiplication (GEMM)
    {
        let gemm_entry_point = match kernel_type {
            ConvKernelType::Im2Col => "im2col_gemm",
            ConvKernelType::Im2ColTiled => "im2col_tiled_gemm",
            _ => unreachable!(),
        };

        let gemm_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("im2col_gemm_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some(gemm_entry_point),
            cache: None,
            compilation_options: Default::default(),
        });

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("im2col_gemm_pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&gemm_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        // Dispatch for GEMM
        let workgroup_x = (out_width + 15) / 16;
        let workgroup_y = (out_height + 15) / 16;
        let workgroup_z = batch_size * out_channels;
        compute_pass.dispatch_workgroups(
            workgroup_x as u32,
            workgroup_y as u32,
            workgroup_z as u32,
        );
    }

    queue.submit(std::iter::once(encoder.finish()));

    Ok(())
}

#[cfg(feature = "gpu")]
#[derive(Debug, Clone, Copy, PartialEq)]
enum ConvKernelType {
    Standard,
    Winograd,
    FFT,
    Tiled,
    Im2Col,
    Im2ColTiled,
}

#[cfg(feature = "gpu")]
fn next_power_of_two(n: usize) -> usize {
    if n <= 1 {
        1
    } else {
        1 << (64 - (n - 1).leading_zeros())
    }
}

#[cfg(feature = "gpu")]
fn select_conv_kernel<T>(
    input_shape: &[usize],
    kernel_shape: &[usize],
    stride: (usize, usize),
    padding: &str,
) -> ConvKernelType {
    let batch_size = input_shape[0];
    let in_channels = input_shape[1];
    let in_height = input_shape[2];
    let in_width = input_shape[3];

    let out_channels = kernel_shape[0];
    let kernel_height = kernel_shape[2];
    let kernel_width = kernel_shape[3];

    // Calculate total operations for heuristic
    let total_input_size = batch_size * in_channels * in_height * in_width;
    let total_kernel_size = kernel_height * kernel_width;

    // Winograd F(2x2, 3x3) is optimal for 3x3 kernels with stride 1
    if kernel_height == 3 && kernel_width == 3 && stride == (1, 1) && padding == "same" {
        // Use Winograd for small to medium sizes where the overhead is worth it
        if total_input_size <= 1024 * 1024 && in_height >= 4 && in_width >= 4 {
            return ConvKernelType::Winograd;
        }
    }

    // FFT-based convolution for large kernels (5x5 or larger)
    if total_kernel_size >= 25 && total_input_size >= 256 * 256 {
        return ConvKernelType::FFT;
    }

    // Im2Col convolution for medium-sized kernels and inputs
    // Best for GEMM-optimized hardware and when we have many channels
    if kernel_height <= 7 && kernel_width <= 7 && in_channels >= 32 && out_channels >= 32 {
        if total_input_size >= 1024 * 1024 {
            return ConvKernelType::Im2ColTiled;
        } else {
            return ConvKernelType::Im2Col;
        }
    }

    // Tiled convolution for memory efficiency with medium sizes
    if total_input_size >= 512 * 512 && total_input_size < 2048 * 2048 {
        return ConvKernelType::Tiled;
    }

    // Default to standard convolution
    ConvKernelType::Standard
}

#[cfg(feature = "gpu")]
fn conv2d_gpu<T>(
    input: &Tensor<T>,
    weight: &Tensor<T>,
    bias: Option<&Tensor<T>>,
    stride: (usize, usize),
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
    let in_height = input_shape.dims()[2];
    let in_width = input_shape.dims()[3];

    let out_channels = weight_shape.dims()[0];
    let kernel_height = weight_shape.dims()[2];
    let kernel_width = weight_shape.dims()[3];

    // Select optimal kernel based on input characteristics
    let kernel_type =
        select_conv_kernel::<T>(input_shape.dims(), weight_shape.dims(), stride, padding);

    // Calculate output dimensions
    let (out_height, out_width, pad_top, pad_left) = match padding {
        "valid" => {
            let out_h = (in_height - kernel_height) / stride.0 + 1;
            let out_w = (in_width - kernel_width) / stride.1 + 1;
            (out_h, out_w, 0, 0)
        }
        "same" => {
            let out_h = (in_height + stride.0 - 1) / stride.0;
            let out_w = (in_width + stride.1 - 1) / stride.1;
            let pad_h = std::cmp::max(0, (out_h - 1) * stride.0 + kernel_height - in_height);
            let pad_w = std::cmp::max(0, (out_w - 1) * stride.1 + kernel_width - in_width);
            (out_h, out_w, pad_h / 2, pad_w / 2)
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
        let output_size = batch_size * out_channels * out_height * out_width;
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("conv2d_output"),
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
            let zero_bias = vec![T::zero(); out_channels];
            let bias_bytes = bytemuck::cast_slice(&zero_bias);
            &(**device).create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("zero_bias"),
                contents: bias_bytes,
                usage: wgpu::BufferUsages::STORAGE,
            })
        };

        // Create parameters uniform buffer
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct ConvParams {
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
        }

        let params = ConvParams {
            batch_size: batch_size as u32,
            in_channels: in_channels as u32,
            input_height: in_height as u32,
            input_width: in_width as u32,
            out_channels: out_channels as u32,
            kernel_height: kernel_height as u32,
            kernel_width: kernel_width as u32,
            output_height: out_height as u32,
            output_width: out_width as u32,
            stride_h: stride.0 as u32,
            stride_w: stride.1 as u32,
            pad_h: pad_top as u32,
            pad_w: pad_left as u32,
        };

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("conv_params"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Load compute shader
        let shader_source = include_str!("../../gpu/shaders/conv_ops.wgsl");
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("conv2d_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("conv2d_bind_group_layout"),
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
            label: Some("conv2d_bind_group"),
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
            label: Some("conv2d_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Select entry point based on kernel type
        let entry_point = match kernel_type {
            ConvKernelType::Standard => "conv2d_kernel",
            ConvKernelType::Winograd => "winograd_input_transform", // Multi-pass Winograd
            ConvKernelType::FFT => "fft_conv_multiply",
            ConvKernelType::Tiled => "tiled_conv2d",
            ConvKernelType::Im2Col => "im2col_gemm",
            ConvKernelType::Im2ColTiled => "im2col_tiled_gemm",
        };

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("conv2d_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some(entry_point),
            cache: None,
            compilation_options: Default::default(),
        });

        // Handle different kernel types
        match kernel_type {
            ConvKernelType::Im2Col | ConvKernelType::Im2ColTiled => {
                // Im2Col requires multi-pass execution: transform + GEMM
                execute_im2col_convolution(
                    device,
                    queue,
                    input_gpu,
                    weight_gpu,
                    &bias_data,
                    &output_buffer,
                    kernel_type,
                    batch_size,
                    in_channels,
                    out_channels,
                    in_height,
                    in_width,
                    out_height,
                    out_width,
                    kernel_height,
                    kernel_width,
                    stride,
                    pad_top,
                    pad_left,
                )?;
            }
            _ => {
                // Standard single-pass execution for other kernel types
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("conv2d_encoder"),
                });

                {
                    let mut compute_pass =
                        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("conv2d_pass"),
                            timestamp_writes: None,
                        });

                    compute_pass.set_pipeline(&compute_pipeline);
                    compute_pass.set_bind_group(0, &bind_group, &[]);

                    // Dispatch with workgroups adapted to kernel type
                    let (workgroup_x, workgroup_y, workgroup_z) = match kernel_type {
                        ConvKernelType::Standard | ConvKernelType::Tiled => {
                            // Standard 2D workgroups for output spatial dimensions
                            let wg_x = (out_width + 7) / 8;
                            let wg_y = (out_height + 7) / 8;
                            let wg_z = batch_size * out_channels;
                            (wg_x, wg_y, wg_z)
                        }
                        ConvKernelType::Winograd => {
                            // Winograd works on 2x2 tiles, so adjust workgroup size
                            let tile_h = (out_height + 1) / 2;
                            let tile_w = (out_width + 1) / 2;
                            let wg_x = (tile_w + 7) / 8;
                            let wg_y = (tile_h + 7) / 8;
                            let wg_z = batch_size * in_channels; // Input channels for transform
                            (wg_x, wg_y, wg_z)
                        }
                        ConvKernelType::FFT => {
                            // FFT works in frequency domain - adjust for FFT size
                            let fft_height = next_power_of_two(in_height + kernel_height - 1);
                            let fft_width = next_power_of_two(in_width + kernel_width - 1);
                            let wg_x = (fft_width + 7) / 8;
                            let wg_y = (fft_height + 7) / 8;
                            let wg_z = batch_size * out_channels;
                            (wg_x, wg_y, wg_z)
                        }
                        _ => unreachable!("Im2Col handled separately"),
                    };

                    compute_pass.dispatch_workgroups(
                        workgroup_x as u32,
                        workgroup_y as u32,
                        workgroup_z as u32,
                    );
                }

                queue.submit(std::iter::once(encoder.finish()));
            }
        }

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
            crate::Shape::new(vec![batch_size, out_channels, out_height, out_width]),
        );
        result.set_requires_grad(input.requires_grad());
        Ok(result)
    } else {
        Err(TensorError::unsupported_operation_simple(
            "GPU convolution requires GPU tensors".to_string(),
        ))
    }
}
