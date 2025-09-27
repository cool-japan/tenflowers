//! Basic pooling operations (max and average pooling) for 2D and 3D tensors
//!
//! This module provides fundamental pooling operations that apply a kernel over
//! spatial dimensions to reduce tensor size while preserving important features.

use crate::tensor::TensorStorage;
#[cfg(feature = "gpu")]
use crate::Shape;
use crate::{Result, Tensor, TensorError};
use num_traits::{Float, FromPrimitive, Zero};
use std::cmp::min;

/// Max pooling 2D operation
/// Input shape: [batch, channels, height, width] (NCHW format) for GPU or [batch, height, width, channels] (NHWC format) for CPU
pub fn max_pool2d<T>(
    input: &Tensor<T>,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: &str,
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + PartialOrd
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    match &input.storage {
        TensorStorage::Cpu(_input_arr) => max_pool2d_cpu(input, kernel_size, stride, padding),
        #[cfg(feature = "gpu")]
        TensorStorage::Gpu(gpu_buffer) => max_pool2d_gpu(input, kernel_size, stride, padding),
    }
}

/// Average pooling 2D operation
/// Input shape: [batch, channels, height, width] (NCHW format) for GPU or [batch, height, width, channels] (NHWC format) for CPU
pub fn avg_pool2d<T>(
    input: &Tensor<T>,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: &str,
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + Float
        + FromPrimitive
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    match &input.storage {
        TensorStorage::Cpu(_input_arr) => avg_pool2d_cpu(input, kernel_size, stride, padding),
        #[cfg(feature = "gpu")]
        TensorStorage::Gpu(gpu_buffer) => avg_pool2d_gpu(input, kernel_size, stride, padding),
    }
}

/// Max pooling 3D operation
/// Input shape: [batch, channels, depth, height, width] (NCDHW format)
pub fn max_pool3d<T>(
    input: &Tensor<T>,
    kernel_size: (usize, usize, usize),
    stride: (usize, usize, usize),
    padding: &str,
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + PartialOrd
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    match &input.storage {
        TensorStorage::Cpu(_input_arr) => max_pool3d_cpu(input, kernel_size, stride, padding),
        #[cfg(feature = "gpu")]
        TensorStorage::Gpu(_gpu_buffer) => max_pool3d_gpu(input, kernel_size, stride, padding),
    }
}

/// Average pooling 3D operation
/// Input shape: [batch, channels, depth, height, width] (NCDHW format)
pub fn avg_pool3d<T>(
    input: &Tensor<T>,
    kernel_size: (usize, usize, usize),
    stride: (usize, usize, usize),
    padding: &str,
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + Float
        + FromPrimitive
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    match &input.storage {
        TensorStorage::Cpu(_input_arr) => avg_pool3d_cpu(input, kernel_size, stride, padding),
        #[cfg(feature = "gpu")]
        TensorStorage::Gpu(_gpu_buffer) => avg_pool3d_gpu(input, kernel_size, stride, padding),
    }
}

// CPU implementations for 2D pooling

fn max_pool2d_cpu<T>(
    input: &Tensor<T>,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: &str,
) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + PartialOrd + Send + Sync + 'static,
{
    let shape = input.shape();
    if shape.rank() != 4 {
        return Err(TensorError::invalid_shape_simple(format!(
            "MaxPool2D expects 4D input, got {}D",
            shape.rank()
        )));
    }

    // CPU assumes NHWC format
    let batch_size = shape.dims()[0];
    let input_height = shape.dims()[1];
    let input_width = shape.dims()[2];
    let channels = shape.dims()[3];

    // Calculate output dimensions
    let (output_height, output_width) = if padding == "valid" {
        (
            (input_height - kernel_size.0) / stride.0 + 1,
            (input_width - kernel_size.1) / stride.1 + 1,
        )
    } else {
        // "same" padding
        (
            (input_height + stride.0 - 1) / stride.0,
            (input_width + stride.1 - 1) / stride.1,
        )
    };

    let mut output_data = vec![T::zero(); batch_size * output_height * output_width * channels];

    for b in 0..batch_size {
        for oh in 0..output_height {
            for ow in 0..output_width {
                for c in 0..channels {
                    let h_start = oh * stride.0;
                    let w_start = ow * stride.1;
                    let h_end = min(h_start + kernel_size.0, input_height);
                    let w_end = min(w_start + kernel_size.1, input_width);

                    let mut max_val: Option<T> = None;
                    for h in h_start..h_end {
                        for w in w_start..w_end {
                            if let Some(val) = input.get(&[b, h, w, c]) {
                                max_val = match max_val {
                                    None => Some(val),
                                    Some(current_max) => {
                                        if val > current_max {
                                            Some(val)
                                        } else {
                                            Some(current_max)
                                        }
                                    }
                                };
                            }
                        }
                    }

                    let out_idx = ((b * output_height + oh) * output_width + ow) * channels + c;
                    output_data[out_idx] = max_val.unwrap_or_else(T::zero);
                }
            }
        }
    }

    Tensor::from_vec(
        output_data,
        &[batch_size, output_height, output_width, channels],
    )
}

fn avg_pool2d_cpu<T>(
    input: &Tensor<T>,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: &str,
) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + Float + FromPrimitive + Send + Sync + 'static,
{
    let shape = input.shape();
    if shape.rank() != 4 {
        return Err(TensorError::invalid_shape_simple(format!(
            "AvgPool2D expects 4D input, got {}D",
            shape.rank()
        )));
    }

    // CPU assumes NHWC format
    let batch_size = shape.dims()[0];
    let input_height = shape.dims()[1];
    let input_width = shape.dims()[2];
    let channels = shape.dims()[3];

    // Calculate output dimensions
    let (output_height, output_width) = if padding == "valid" {
        (
            (input_height - kernel_size.0) / stride.0 + 1,
            (input_width - kernel_size.1) / stride.1 + 1,
        )
    } else {
        // "same" padding
        (
            (input_height + stride.0 - 1) / stride.0,
            (input_width + stride.1 - 1) / stride.1,
        )
    };

    let mut output_data = vec![T::zero(); batch_size * output_height * output_width * channels];

    for b in 0..batch_size {
        for oh in 0..output_height {
            for ow in 0..output_width {
                for c in 0..channels {
                    let h_start = oh * stride.0;
                    let w_start = ow * stride.1;
                    let h_end = min(h_start + kernel_size.0, input_height);
                    let w_end = min(w_start + kernel_size.1, input_width);

                    let mut sum = T::zero();
                    let mut count = 0;

                    for h in h_start..h_end {
                        for w in w_start..w_end {
                            if let Some(val) = input.get(&[b, h, w, c]) {
                                sum = sum + val;
                                count += 1;
                            }
                        }
                    }

                    let out_idx = ((b * output_height + oh) * output_width + ow) * channels + c;
                    if count > 0 {
                        output_data[out_idx] = sum / T::from(count).unwrap();
                    } else {
                        output_data[out_idx] = T::zero();
                    }
                }
            }
        }
    }

    Tensor::from_vec(
        output_data,
        &[batch_size, output_height, output_width, channels],
    )
}

// CPU implementations for 3D pooling

fn max_pool3d_cpu<T>(
    input: &Tensor<T>,
    kernel_size: (usize, usize, usize),
    stride: (usize, usize, usize),
    padding: &str,
) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + PartialOrd + Send + Sync + 'static,
{
    let shape = input.shape();
    if shape.rank() != 5 {
        return Err(TensorError::InvalidShape {
            operation: "max_pool3d".to_string(),
            reason: format!("MaxPool3D expects 5D input, got {}D", shape.rank()),
            shape: Some(shape.dims().to_vec()),
            context: None,
        });
    }

    // CPU assumes NCDHW format
    let batch_size = shape.dims()[0];
    let channels = shape.dims()[1];
    let input_depth = shape.dims()[2];
    let input_height = shape.dims()[3];
    let input_width = shape.dims()[4];

    // Calculate output dimensions
    let (output_depth, output_height, output_width) = if padding == "valid" {
        (
            (input_depth - kernel_size.0) / stride.0 + 1,
            (input_height - kernel_size.1) / stride.1 + 1,
            (input_width - kernel_size.2) / stride.2 + 1,
        )
    } else {
        // "same" padding
        (
            (input_depth + stride.0 - 1) / stride.0,
            (input_height + stride.1 - 1) / stride.1,
            (input_width + stride.2 - 1) / stride.2,
        )
    };

    let mut output_data =
        vec![T::zero(); batch_size * channels * output_depth * output_height * output_width];

    for b in 0..batch_size {
        for c in 0..channels {
            for od in 0..output_depth {
                for oh in 0..output_height {
                    for ow in 0..output_width {
                        let d_start = od * stride.0;
                        let h_start = oh * stride.1;
                        let w_start = ow * stride.2;
                        let d_end = min(d_start + kernel_size.0, input_depth);
                        let h_end = min(h_start + kernel_size.1, input_height);
                        let w_end = min(w_start + kernel_size.2, input_width);

                        let mut max_val: Option<T> = None;
                        for d in d_start..d_end {
                            for h in h_start..h_end {
                                for w in w_start..w_end {
                                    if let Some(val) = input.get(&[b, c, d, h, w]) {
                                        max_val = match max_val {
                                            None => Some(val),
                                            Some(current_max) => {
                                                if val > current_max {
                                                    Some(val)
                                                } else {
                                                    Some(current_max)
                                                }
                                            }
                                        };
                                    }
                                }
                            }
                        }

                        let out_idx = b * channels * output_depth * output_height * output_width
                            + c * output_depth * output_height * output_width
                            + od * output_height * output_width
                            + oh * output_width
                            + ow;
                        output_data[out_idx] = max_val.unwrap_or_else(T::zero);
                    }
                }
            }
        }
    }

    Tensor::from_vec(
        output_data,
        &[
            batch_size,
            channels,
            output_depth,
            output_height,
            output_width,
        ],
    )
}

fn avg_pool3d_cpu<T>(
    input: &Tensor<T>,
    kernel_size: (usize, usize, usize),
    stride: (usize, usize, usize),
    padding: &str,
) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + Float + FromPrimitive + Send + Sync + 'static,
{
    let shape = input.shape();
    if shape.rank() != 5 {
        return Err(TensorError::invalid_shape_simple(format!(
            "AvgPool3D expects 5D input, got {}D",
            shape.rank()
        )));
    }

    // CPU assumes NCDHW format
    let batch_size = shape.dims()[0];
    let channels = shape.dims()[1];
    let input_depth = shape.dims()[2];
    let input_height = shape.dims()[3];
    let input_width = shape.dims()[4];

    // Calculate output dimensions
    let (output_depth, output_height, output_width) = if padding == "valid" {
        (
            (input_depth - kernel_size.0) / stride.0 + 1,
            (input_height - kernel_size.1) / stride.1 + 1,
            (input_width - kernel_size.2) / stride.2 + 1,
        )
    } else {
        // "same" padding
        (
            (input_depth + stride.0 - 1) / stride.0,
            (input_height + stride.1 - 1) / stride.1,
            (input_width + stride.2 - 1) / stride.2,
        )
    };

    let mut output_data =
        vec![T::zero(); batch_size * channels * output_depth * output_height * output_width];

    for b in 0..batch_size {
        for c in 0..channels {
            for od in 0..output_depth {
                for oh in 0..output_height {
                    for ow in 0..output_width {
                        let d_start = od * stride.0;
                        let h_start = oh * stride.1;
                        let w_start = ow * stride.2;
                        let d_end = min(d_start + kernel_size.0, input_depth);
                        let h_end = min(h_start + kernel_size.1, input_height);
                        let w_end = min(w_start + kernel_size.2, input_width);

                        let mut sum = T::zero();
                        let mut count = 0;

                        for d in d_start..d_end {
                            for h in h_start..h_end {
                                for w in w_start..w_end {
                                    if let Some(val) = input.get(&[b, c, d, h, w]) {
                                        sum = sum + val;
                                        count += 1;
                                    }
                                }
                            }
                        }

                        let out_idx = b * channels * output_depth * output_height * output_width
                            + c * output_depth * output_height * output_width
                            + od * output_height * output_width
                            + oh * output_width
                            + ow;
                        if count > 0 {
                            output_data[out_idx] = sum / T::from(count).unwrap();
                        } else {
                            output_data[out_idx] = T::zero();
                        }
                    }
                }
            }
        }
    }

    Tensor::from_vec(
        output_data,
        &[
            batch_size,
            channels,
            output_depth,
            output_height,
            output_width,
        ],
    )
}

// GPU implementations for 2D pooling

#[cfg(feature = "gpu")]
fn max_pool2d_gpu<T>(
    input: &Tensor<T>,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: &str,
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + PartialOrd
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    use crate::gpu::buffer::GpuBuffer;

    let shape = input.shape();
    if shape.rank() != 4 {
        return Err(TensorError::invalid_shape_simple(format!(
            "MaxPool2D expects 4D input, got {}D",
            shape.rank()
        )));
    }

    // GPU assumes NCHW format
    let batch_size = shape.dims()[0];
    let channels = shape.dims()[1];
    let input_height = shape.dims()[2];
    let input_width = shape.dims()[3];

    // Calculate output dimensions
    let (output_height, output_width) = if padding == "valid" {
        (
            (input_height - kernel_size.0) / stride.0 + 1,
            (input_width - kernel_size.1) / stride.1 + 1,
        )
    } else {
        // "same" padding
        (
            (input_height + stride.0 - 1) / stride.0,
            (input_width + stride.1 - 1) / stride.1,
        )
    };

    let input_shape = &[batch_size, channels, input_height, input_width];
    let output_shape = &[batch_size, channels, output_height, output_width];
    let padding_tuple = if padding == "same" {
        let pad_h = std::cmp::max(
            0,
            (output_height - 1) * stride.0 + kernel_size.0 - input_height,
        ) / 2;
        let pad_w = std::cmp::max(
            0,
            (output_width - 1) * stride.1 + kernel_size.1 - input_width,
        ) / 2;
        (pad_h, pad_w)
    } else {
        (0, 0)
    };

    let TensorStorage::Gpu(gpu_buffer) = &input.storage else {
        return Err(TensorError::unsupported_operation_simple(
            "Internal error: max_pool2d_gpu called with non-GPU tensor".to_string(),
        ));
    };

    let kernel_size_slice = &[kernel_size.0, kernel_size.1];
    let stride_slice = &[stride.0, stride.1];
    let padding_slice = &[padding_tuple.0, padding_tuple.1];
    let output_len = output_shape.iter().product();

    let result_gpu = crate::gpu::ops::execute_pooling_op(
        gpu_buffer,
        crate::gpu::ops::PoolingOp::MaxPool2D,
        kernel_size_slice,
        stride_slice,
        padding_slice,
        input_shape,
        output_len,
    )?;

    let mut result = Tensor::from_gpu_buffer(result_gpu, crate::Shape::new(output_shape.to_vec()));
    result.set_requires_grad(input.requires_grad());
    Ok(result)
}

#[cfg(feature = "gpu")]
fn avg_pool2d_gpu<T>(
    input: &Tensor<T>,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: &str,
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + Float
        + FromPrimitive
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    use crate::gpu::buffer::GpuBuffer;

    let shape = input.shape();
    if shape.rank() != 4 {
        return Err(TensorError::invalid_shape_simple(format!(
            "AvgPool2D expects 4D input, got {}D",
            shape.rank()
        )));
    }

    // GPU assumes NCHW format
    let batch_size = shape.dims()[0];
    let channels = shape.dims()[1];
    let input_height = shape.dims()[2];
    let input_width = shape.dims()[3];

    // Calculate output dimensions
    let (output_height, output_width) = if padding == "valid" {
        (
            (input_height - kernel_size.0) / stride.0 + 1,
            (input_width - kernel_size.1) / stride.1 + 1,
        )
    } else {
        // "same" padding
        (
            (input_height + stride.0 - 1) / stride.0,
            (input_width + stride.1 - 1) / stride.1,
        )
    };

    let input_shape = &[batch_size, channels, input_height, input_width];
    let output_shape = &[batch_size, channels, output_height, output_width];
    let padding_tuple = if padding == "same" {
        let pad_h = std::cmp::max(
            0,
            (output_height - 1) * stride.0 + kernel_size.0 - input_height,
        ) / 2;
        let pad_w = std::cmp::max(
            0,
            (output_width - 1) * stride.1 + kernel_size.1 - input_width,
        ) / 2;
        (pad_h, pad_w)
    } else {
        (0, 0)
    };

    let TensorStorage::Gpu(gpu_buffer) = &input.storage else {
        return Err(TensorError::unsupported_operation_simple(
            "Internal error: avg_pool2d_gpu called with non-GPU tensor".to_string(),
        ));
    };

    let kernel_size_slice = &[kernel_size.0, kernel_size.1];
    let stride_slice = &[stride.0, stride.1];
    let padding_slice = &[padding_tuple.0, padding_tuple.1];
    let output_len = output_shape.iter().product();

    let result_gpu = crate::gpu::ops::execute_pooling_op(
        gpu_buffer,
        crate::gpu::ops::PoolingOp::AvgPool2D,
        kernel_size_slice,
        stride_slice,
        padding_slice,
        input_shape,
        output_len,
    )?;

    let mut result = Tensor::from_gpu_buffer(result_gpu, crate::Shape::new(output_shape.to_vec()));
    result.set_requires_grad(input.requires_grad());
    Ok(result)
}

// GPU implementations for 3D pooling

#[cfg(feature = "gpu")]
fn max_pool3d_gpu<T>(
    input: &Tensor<T>,
    kernel_size: (usize, usize, usize),
    stride: (usize, usize, usize),
    padding: &str,
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + PartialOrd
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    let shape = input.shape();
    if shape.rank() != 5 {
        return Err(TensorError::InvalidShape {
            operation: "max_pool3d".to_string(),
            reason: format!("MaxPool3D expects 5D input, got {}D", shape.rank()),
            shape: Some(shape.dims().to_vec()),
            context: None,
        });
    }

    // GPU assumes NCDHW format
    let batch_size = shape.dims()[0];
    let channels = shape.dims()[1];
    let input_depth = shape.dims()[2];
    let input_height = shape.dims()[3];
    let input_width = shape.dims()[4];

    // Calculate output dimensions
    let (output_depth, output_height, output_width) = if padding == "valid" {
        (
            (input_depth - kernel_size.0) / stride.0 + 1,
            (input_height - kernel_size.1) / stride.1 + 1,
            (input_width - kernel_size.2) / stride.2 + 1,
        )
    } else {
        // "same" padding
        (
            (input_depth + stride.0 - 1) / stride.0,
            (input_height + stride.1 - 1) / stride.1,
            (input_width + stride.2 - 1) / stride.2,
        )
    };

    let input_shape = &[batch_size, channels, input_depth, input_height, input_width];
    let output_shape = &[
        batch_size,
        channels,
        output_depth,
        output_height,
        output_width,
    ];

    let TensorStorage::Gpu(gpu_buffer) = &input.storage else {
        return Err(TensorError::unsupported_operation_simple(
            "Internal error: max_pool3d_gpu called with non-GPU tensor".to_string(),
        ));
    };

    let kernel_size_slice = &[kernel_size.0, kernel_size.1];
    let stride_slice = &[stride.0, stride.1];
    let padding_slice = &[0, 0];
    let output_len = output_shape.iter().product();

    let result_gpu = crate::gpu::ops::execute_pooling_op(
        gpu_buffer,
        crate::gpu::ops::PoolingOp::MaxPool3D,
        kernel_size_slice,
        stride_slice,
        padding_slice,
        input_shape,
        output_len,
    )?;

    let mut result = Tensor::from_gpu_buffer(result_gpu, crate::Shape::new(output_shape.to_vec()));
    result.set_requires_grad(input.requires_grad());
    Ok(result)
}

#[cfg(feature = "gpu")]
fn avg_pool3d_gpu<T>(
    input: &Tensor<T>,
    kernel_size: (usize, usize, usize),
    stride: (usize, usize, usize),
    padding: &str,
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + Float
        + FromPrimitive
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    let shape = input.shape();
    if shape.rank() != 5 {
        return Err(TensorError::invalid_shape_simple(format!(
            "AvgPool3D expects 5D input, got {}D",
            shape.rank()
        )));
    }

    // GPU assumes NCDHW format
    let batch_size = shape.dims()[0];
    let channels = shape.dims()[1];
    let input_depth = shape.dims()[2];
    let input_height = shape.dims()[3];
    let input_width = shape.dims()[4];

    // Calculate output dimensions
    let (output_depth, output_height, output_width) = if padding == "valid" {
        (
            (input_depth - kernel_size.0) / stride.0 + 1,
            (input_height - kernel_size.1) / stride.1 + 1,
            (input_width - kernel_size.2) / stride.2 + 1,
        )
    } else {
        // "same" padding
        (
            (input_depth + stride.0 - 1) / stride.0,
            (input_height + stride.1 - 1) / stride.1,
            (input_width + stride.2 - 1) / stride.2,
        )
    };

    let input_shape = &[batch_size, channels, input_depth, input_height, input_width];
    let output_shape = &[
        batch_size,
        channels,
        output_depth,
        output_height,
        output_width,
    ];

    let TensorStorage::Gpu(gpu_buffer) = &input.storage else {
        return Err(TensorError::unsupported_operation_simple(
            "Internal error: avg_pool3d_gpu called with non-GPU tensor".to_string(),
        ));
    };

    let kernel_size_slice = &[kernel_size.0, kernel_size.1];
    let stride_slice = &[stride.0, stride.1];
    let padding_slice = &[0, 0];
    let output_len = output_shape.iter().product();

    let result_gpu = crate::gpu::ops::execute_pooling_op(
        gpu_buffer,
        crate::gpu::ops::PoolingOp::AvgPool3D,
        kernel_size_slice,
        stride_slice,
        padding_slice,
        input_shape,
        output_len,
    )?;

    let mut result = Tensor::from_gpu_buffer(result_gpu, crate::Shape::new(output_shape.to_vec()));
    result.set_requires_grad(input.requires_grad());
    Ok(result)
}
