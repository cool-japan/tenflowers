//! Global pooling operations for 2D and 3D tensors
//!
//! Global pooling operations reduce spatial dimensions to size 1 by pooling
//! over the entire spatial extent. These are commonly used before final
//! classification layers to create feature vectors.

use crate::tensor::TensorStorage;
use crate::{Result, Tensor, TensorError};
use scirs2_core::numeric::{Float, FromPrimitive, Zero};

/// Global max pooling 2D - pools over the entire spatial dimensions
/// Input shape: [batch, channels, height, width] (NCHW format)
/// Output shape: [batch, channels, 1, 1]
pub fn global_max_pool2d<T>(input: &Tensor<T>) -> Result<Tensor<T>>
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
        TensorStorage::Cpu(_input_arr) => global_max_pool2d_cpu(input),
        #[cfg(feature = "gpu")]
        TensorStorage::Gpu(_gpu_buffer) => global_max_pool2d_gpu(input),
    }
}

/// Global average pooling 2D - pools over the entire spatial dimensions
/// Input shape: [batch, channels, height, width] (NCHW format)
/// Output shape: [batch, channels, 1, 1]
pub fn global_avg_pool2d<T>(input: &Tensor<T>) -> Result<Tensor<T>>
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
        TensorStorage::Cpu(_input_arr) => global_avg_pool2d_cpu(input),
        #[cfg(feature = "gpu")]
        TensorStorage::Gpu(_gpu_buffer) => global_avg_pool2d_gpu(input),
    }
}

/// Global max pooling 3D - pools over the entire spatial dimensions
/// Input shape: [batch, channels, depth, height, width] (NCDHW format)
/// Output shape: [batch, channels, 1, 1, 1]
pub fn global_max_pool3d<T>(input: &Tensor<T>) -> Result<Tensor<T>>
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
        TensorStorage::Cpu(_input_arr) => global_max_pool3d_cpu(input),
        #[cfg(feature = "gpu")]
        TensorStorage::Gpu(_gpu_buffer) => global_max_pool3d_gpu(input),
    }
}

/// Global average pooling 3D - pools over the entire spatial dimensions
/// Input shape: [batch, channels, depth, height, width] (NCDHW format)
/// Output shape: [batch, channels, 1, 1, 1]
pub fn global_avg_pool3d<T>(input: &Tensor<T>) -> Result<Tensor<T>>
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
        TensorStorage::Cpu(_input_arr) => global_avg_pool3d_cpu(input),
        #[cfg(feature = "gpu")]
        TensorStorage::Gpu(_gpu_buffer) => global_avg_pool3d_gpu(input),
    }
}

// CPU implementations for 2D global pooling

#[allow(clippy::infallible_destructuring_match)]
fn global_max_pool2d_cpu<T>(input: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + PartialOrd,
{
    let input_arr = match &input.storage {
        TensorStorage::Cpu(arr) => arr,
        #[cfg(feature = "gpu")]
        TensorStorage::Gpu(_) => {
            panic!("global_max_pool2d_cpu should only be called with CPU tensors")
        }
    };

    if input_arr.ndim() != 4 {
        return Err(TensorError::invalid_shape_simple(
            "Global max pool input must be 4D (NCHW format)".to_string(),
        ));
    }

    let input_shape = input_arr.shape();
    let batch_size = input_shape[0];
    let channels = input_shape[1];
    let height = input_shape[2];
    let width = input_shape[3];

    let output_shape = vec![batch_size, channels, 1, 1];
    let mut output_data = vec![T::default(); batch_size * channels];

    for b in 0..batch_size {
        for c in 0..channels {
            let mut max_val = T::default();
            let mut has_val = false;

            for h in 0..height {
                for w in 0..width {
                    let val = input_arr[[b, c, h, w]].clone();
                    if !has_val || val > max_val {
                        max_val = val;
                        has_val = true;
                    }
                }
            }

            output_data[b * channels + c] = max_val;
        }
    }

    Tensor::from_vec(output_data, &output_shape)
}

#[allow(clippy::infallible_destructuring_match)]
fn global_avg_pool2d_cpu<T>(input: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + Float + FromPrimitive,
{
    let input_arr = match &input.storage {
        TensorStorage::Cpu(arr) => arr,
        #[cfg(feature = "gpu")]
        TensorStorage::Gpu(_) => {
            panic!("adaptive_avg_pool2d_cpu should only be called with CPU tensors")
        }
    };

    if input_arr.ndim() != 4 {
        return Err(TensorError::invalid_shape_simple(
            "Global avg pool input must be 4D (NCHW format)".to_string(),
        ));
    }

    let input_shape = input_arr.shape();
    let batch_size = input_shape[0];
    let channels = input_shape[1];
    let height = input_shape[2];
    let width = input_shape[3];

    let output_shape = vec![batch_size, channels, 1, 1];
    let mut output_data = vec![T::default(); batch_size * channels];
    let spatial_size = T::from_usize(height * width).unwrap_or(T::one());

    for b in 0..batch_size {
        for c in 0..channels {
            let mut sum = T::zero();

            for h in 0..height {
                for w in 0..width {
                    sum = sum + input_arr[[b, c, h, w]];
                }
            }

            output_data[b * channels + c] = sum / spatial_size;
        }
    }

    Tensor::from_vec(output_data, &output_shape)
}

// CPU implementations for 3D global pooling

#[allow(clippy::infallible_destructuring_match)]
fn global_max_pool3d_cpu<T>(input: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + PartialOrd,
{
    let input_arr = match &input.storage {
        TensorStorage::Cpu(arr) => arr,
        #[cfg(feature = "gpu")]
        TensorStorage::Gpu(_) => {
            panic!("global_max_pool3d_cpu should only be called with CPU tensors")
        }
    };

    if input_arr.ndim() != 5 {
        return Err(TensorError::invalid_shape_simple(
            "Global max pool 3D input must be 5D (NCDHW format)".to_string(),
        ));
    }

    let input_shape = input_arr.shape();
    let batch_size = input_shape[0];
    let channels = input_shape[1];
    let depth = input_shape[2];
    let height = input_shape[3];
    let width = input_shape[4];

    let output_shape = vec![batch_size, channels, 1, 1, 1];
    let mut output_data = vec![T::default(); batch_size * channels];

    for b in 0..batch_size {
        for c in 0..channels {
            let mut max_val = T::default();
            let mut has_val = false;

            for d in 0..depth {
                for h in 0..height {
                    for w in 0..width {
                        let val = input_arr[[b, c, d, h, w]].clone();
                        if !has_val || val > max_val {
                            max_val = val;
                            has_val = true;
                        }
                    }
                }
            }

            output_data[b * channels + c] = max_val;
        }
    }

    Tensor::from_vec(output_data, &output_shape)
}

#[allow(clippy::infallible_destructuring_match)]
fn global_avg_pool3d_cpu<T>(input: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + Float + FromPrimitive,
{
    let input_arr = match &input.storage {
        TensorStorage::Cpu(arr) => arr,
        #[cfg(feature = "gpu")]
        TensorStorage::Gpu(_) => {
            panic!("global_avg_pool3d_cpu should only be called with CPU tensors")
        }
    };

    if input_arr.ndim() != 5 {
        return Err(TensorError::invalid_shape_simple(
            "Global avg pool 3D input must be 5D (NCDHW format)".to_string(),
        ));
    }

    let input_shape = input_arr.shape();
    let batch_size = input_shape[0];
    let channels = input_shape[1];
    let depth = input_shape[2];
    let height = input_shape[3];
    let width = input_shape[4];

    let output_shape = vec![batch_size, channels, 1, 1, 1];
    let mut output_data = vec![T::default(); batch_size * channels];
    let spatial_size = T::from_usize(depth * height * width).unwrap_or(T::one());

    for b in 0..batch_size {
        for c in 0..channels {
            let mut sum = T::zero();

            for d in 0..depth {
                for h in 0..height {
                    for w in 0..width {
                        sum = sum + input_arr[[b, c, d, h, w]];
                    }
                }
            }

            output_data[b * channels + c] = sum / spatial_size;
        }
    }

    Tensor::from_vec(output_data, &output_shape)
}

// GPU implementations for 2D global pooling

#[cfg(feature = "gpu")]
fn global_max_pool2d_gpu<T>(input: &Tensor<T>) -> Result<Tensor<T>>
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
    let TensorStorage::Gpu(gpu_buffer) = &input.storage else {
        return Err(TensorError::unsupported_operation_simple(
            "Internal error: global_max_pool2d_gpu called with non-GPU tensor".to_string(),
        ));
    };

    let input_shape = input.shape().dims();
    if input_shape.len() != 4 {
        return Err(TensorError::invalid_shape_simple(
            "Global max pool input must be 4D (NCHW format)".to_string(),
        ));
    }

    let batch_size = input_shape[0];
    let channels = input_shape[1];
    let output_shape = vec![batch_size, channels, 1, 1];

    let kernel_size = &[input_shape[2], input_shape[3]];
    let stride = &[1, 1];
    let padding = &[0, 0];
    let output_len = output_shape.iter().product();

    let result_gpu = crate::gpu::ops::execute_pooling_op(
        gpu_buffer,
        crate::gpu::ops::PoolingOp::GlobalMaxPool,
        kernel_size,
        stride,
        padding,
        input_shape,
        output_len,
    )?;

    let mut result = Tensor::from_gpu_buffer(result_gpu, crate::Shape::new(output_shape));
    result.set_requires_grad(input.requires_grad());
    Ok(result)
}

#[cfg(feature = "gpu")]
fn global_avg_pool2d_gpu<T>(input: &Tensor<T>) -> Result<Tensor<T>>
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
    let TensorStorage::Gpu(gpu_buffer) = &input.storage else {
        return Err(TensorError::unsupported_operation_simple(
            "Internal error: global_avg_pool2d_gpu called with non-GPU tensor".to_string(),
        ));
    };

    let input_shape = input.shape().dims();
    if input_shape.len() != 4 {
        return Err(TensorError::invalid_shape_simple(
            "Global avg pool input must be 4D (NCHW format)".to_string(),
        ));
    }

    let batch_size = input_shape[0];
    let channels = input_shape[1];
    let output_shape = vec![batch_size, channels, 1, 1];

    let kernel_size = &[input_shape[2], input_shape[3]];
    let stride = &[1, 1];
    let padding = &[0, 0];
    let output_len = output_shape.iter().product();

    let result_gpu = crate::gpu::ops::execute_pooling_op(
        gpu_buffer,
        crate::gpu::ops::PoolingOp::GlobalAvgPool,
        kernel_size,
        stride,
        padding,
        input_shape,
        output_len,
    )?;

    let mut result = Tensor::from_gpu_buffer(result_gpu, crate::Shape::new(output_shape));
    result.set_requires_grad(input.requires_grad());
    Ok(result)
}

// GPU implementations for 3D global pooling

#[cfg(feature = "gpu")]
fn global_max_pool3d_gpu<T>(input: &Tensor<T>) -> Result<Tensor<T>>
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
    let cpu_input = input.to_cpu()?;
    let result = global_max_pool3d_cpu(&cpu_input)?;
    result.to_device(input.device().clone())
}

#[cfg(feature = "gpu")]
fn global_avg_pool3d_gpu<T>(input: &Tensor<T>) -> Result<Tensor<T>>
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
    let cpu_input = input.to_cpu()?;
    let result = global_avg_pool3d_cpu(&cpu_input)?;
    result.to_device(input.device().clone())
}

// GPU delegate correctness tests: verify global_max_pool3d_gpu/
// global_avg_pool3d_gpu round-trip GPU-resident input through the host and
// delegate to the known-correct CPU implementation, instead of erroring via
// execute_pooling_op (which has no GlobalMaxPool3D/GlobalAvgPool3D
// shader_entry_point arm — only the 2D GlobalMaxPool/GlobalAvgPool variants
// are native). global_max_pool2d_gpu/global_avg_pool2d_gpu (2D) already have
// real native GPU kernels and are intentionally not touched or tested here.
#[cfg(all(test, feature = "gpu"))]
mod gpu_delegate_tests {
    use super::*;
    use crate::Device;

    #[test]
    fn gpu_global_max_pool3d_matches_cpu_reference() {
        let data: Vec<f32> = (1..=8).map(|v| v as f32).collect();
        let cpu_input =
            Tensor::<f32>::from_vec(data, &[1, 1, 2, 2, 2]).expect("test: from_vec should succeed");
        let cpu_result =
            global_max_pool3d(&cpu_input).expect("test: CPU global_max_pool3d should succeed");

        let gpu_input = match cpu_input.to(Device::Gpu(0)) {
            Ok(t) => t,
            Err(_) => return, // No GPU adapter available in this environment; skip.
        };
        let gpu_result = global_max_pool3d(&gpu_input)
            .expect("test: GPU global_max_pool3d should succeed via CPU-delegate fallback");
        assert_eq!(gpu_result.shape().dims(), cpu_result.shape().dims());
        assert_eq!(
            gpu_result.to_vec().expect("test: to_vec should succeed"),
            cpu_result.to_vec().expect("test: to_vec should succeed"),
        );
    }

    #[test]
    fn gpu_global_avg_pool3d_matches_cpu_reference() {
        let data: Vec<f32> = (1..=8).map(|v| v as f32).collect();
        let cpu_input =
            Tensor::<f32>::from_vec(data, &[1, 1, 2, 2, 2]).expect("test: from_vec should succeed");
        let cpu_result =
            global_avg_pool3d(&cpu_input).expect("test: CPU global_avg_pool3d should succeed");

        let gpu_input = match cpu_input.to(Device::Gpu(0)) {
            Ok(t) => t,
            Err(_) => return,
        };
        let gpu_result = global_avg_pool3d(&gpu_input)
            .expect("test: GPU global_avg_pool3d should succeed via CPU-delegate fallback");
        assert_eq!(gpu_result.shape().dims(), cpu_result.shape().dims());
        assert_eq!(
            gpu_result.to_vec().expect("test: to_vec should succeed"),
            cpu_result.to_vec().expect("test: to_vec should succeed"),
        );
    }
}
