//! Adaptive pooling operations for 2D tensors
//!
//! Adaptive pooling allows pooling to a specific output size regardless of input size,
//! commonly used to ensure consistent output dimensions for different input sizes.

use crate::tensor::TensorStorage;
use crate::{Result, Tensor, TensorError};
use scirs2_core::numeric::{Float, FromPrimitive, Zero};
use std::cmp::min;

/// Adaptive average pooling 2D - pools to a specific output size
/// Input shape: [batch, channels, height, width] (NCHW format)
/// Output shape: [batch, channels, output_height, output_width]
pub fn adaptive_avg_pool2d<T>(input: &Tensor<T>, output_size: (usize, usize)) -> Result<Tensor<T>>
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
        TensorStorage::Cpu(_input_arr) => adaptive_avg_pool2d_cpu(input, output_size),
        #[cfg(feature = "gpu")]
        TensorStorage::Gpu(_gpu_buffer) => adaptive_avg_pool2d_gpu(input, output_size),
    }
}

/// Adaptive max pooling 2D - pools to a specific output size
/// Input shape: [batch, channels, height, width] (NCHW format)
/// Output shape: [batch, channels, output_height, output_width]
pub fn adaptive_max_pool2d<T>(input: &Tensor<T>, output_size: (usize, usize)) -> Result<Tensor<T>>
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
        TensorStorage::Cpu(_input_arr) => adaptive_max_pool2d_cpu(input, output_size),
        #[cfg(feature = "gpu")]
        TensorStorage::Gpu(_gpu_buffer) => adaptive_max_pool2d_gpu(input, output_size),
    }
}

// CPU implementations

#[allow(clippy::infallible_destructuring_match)]
fn adaptive_avg_pool2d_cpu<T>(input: &Tensor<T>, output_size: (usize, usize)) -> Result<Tensor<T>>
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
            "Adaptive avg pool input must be 4D (NCHW format)".to_string(),
        ));
    }

    let input_shape = input_arr.shape();
    let batch_size = input_shape[0];
    let channels = input_shape[1];
    let in_height = input_shape[2];
    let in_width = input_shape[3];

    let (out_height, out_width) = output_size;
    let output_shape = vec![batch_size, channels, out_height, out_width];
    let mut output_data = vec![T::default(); batch_size * channels * out_height * out_width];

    for b in 0..batch_size {
        for c in 0..channels {
            for out_h in 0..out_height {
                for out_w in 0..out_width {
                    // Calculate adaptive pooling window
                    let h_start = (out_h * in_height) / out_height;
                    let h_end = ((out_h + 1) * in_height + out_height - 1) / out_height;
                    let w_start = (out_w * in_width) / out_width;
                    let w_end = ((out_w + 1) * in_width + out_width - 1) / out_width;

                    let h_end = min(h_end, in_height);
                    let w_end = min(w_end, in_width);

                    let mut sum = T::zero();
                    let mut count = 0;

                    for h in h_start..h_end {
                        for w in w_start..w_end {
                            sum = sum + input_arr[[b, c, h, w]];
                            count += 1;
                        }
                    }

                    let avg = if count > 0 {
                        sum / T::from_usize(count).unwrap_or(T::one())
                    } else {
                        T::zero()
                    };

                    let output_idx = b * channels * out_height * out_width
                        + c * out_height * out_width
                        + out_h * out_width
                        + out_w;
                    output_data[output_idx] = avg;
                }
            }
        }
    }

    Tensor::from_vec(output_data, &output_shape)
}

#[allow(clippy::infallible_destructuring_match)]
fn adaptive_max_pool2d_cpu<T>(input: &Tensor<T>, output_size: (usize, usize)) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + PartialOrd + Send + Sync + 'static,
{
    let input_arr = match &input.storage {
        TensorStorage::Cpu(arr) => arr,
        #[cfg(feature = "gpu")]
        TensorStorage::Gpu(_) => {
            panic!("adaptive_max_pool2d_cpu should only be called with CPU tensors")
        }
    };

    if input_arr.ndim() != 4 {
        return Err(TensorError::invalid_shape_simple(
            "Adaptive max pool input must be 4D (NCHW format)".to_string(),
        ));
    }

    let input_shape = input_arr.shape();
    let batch_size = input_shape[0];
    let channels = input_shape[1];
    let in_height = input_shape[2];
    let in_width = input_shape[3];

    let (out_height, out_width) = output_size;
    let output_shape = vec![batch_size, channels, out_height, out_width];
    let mut output_data = vec![T::default(); batch_size * channels * out_height * out_width];

    for b in 0..batch_size {
        for c in 0..channels {
            for out_h in 0..out_height {
                for out_w in 0..out_width {
                    // Calculate adaptive pooling window
                    let h_start = (out_h * in_height) / out_height;
                    let h_end = ((out_h + 1) * in_height + out_height - 1) / out_height;
                    let w_start = (out_w * in_width) / out_width;
                    let w_end = ((out_w + 1) * in_width + out_width - 1) / out_width;

                    let h_end = min(h_end, in_height);
                    let w_end = min(w_end, in_width);

                    let mut max_val: Option<T> = None;

                    for h in h_start..h_end {
                        for w in w_start..w_end {
                            let val = input_arr[[b, c, h, w]].clone();
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

                    let output_idx = b * channels * out_height * out_width
                        + c * out_height * out_width
                        + out_h * out_width
                        + out_w;
                    output_data[output_idx] = max_val.unwrap_or_else(T::zero);
                }
            }
        }
    }

    Tensor::from_vec(output_data, &output_shape)
}

// GPU implementations

#[cfg(feature = "gpu")]
fn adaptive_avg_pool2d_gpu<T>(input: &Tensor<T>, output_size: (usize, usize)) -> Result<Tensor<T>>
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
    let result = adaptive_avg_pool2d_cpu(&cpu_input, output_size)?;
    result.to_device(input.device().clone())
}

#[cfg(feature = "gpu")]
fn adaptive_max_pool2d_gpu<T>(input: &Tensor<T>, output_size: (usize, usize)) -> Result<Tensor<T>>
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
    let result = adaptive_max_pool2d_cpu(&cpu_input, output_size)?;
    result.to_device(input.device().clone())
}

// GPU delegate correctness tests: verify adaptive_avg_pool2d_gpu/
// adaptive_max_pool2d_gpu round-trip GPU-resident input through the host and
// delegate to the known-correct CPU implementation, instead of erroring via
// execute_pooling_op (which has no AdaptiveAvgPool2D/AdaptiveMaxPool2D
// shader_entry_point arm).
#[cfg(all(test, feature = "gpu"))]
mod gpu_delegate_tests {
    use super::*;
    use crate::Device;

    #[test]
    fn gpu_adaptive_avg_pool2d_matches_cpu_reference() {
        let data: Vec<f32> = (0..16).map(|v| v as f32).collect();
        let cpu_input =
            Tensor::<f32>::from_vec(data, &[1, 1, 4, 4]).expect("test: from_vec should succeed");
        let cpu_result = adaptive_avg_pool2d(&cpu_input, (2, 2))
            .expect("test: CPU adaptive_avg_pool2d should succeed");

        let gpu_input = match cpu_input.to(Device::Gpu(0)) {
            Ok(t) => t,
            Err(_) => return, // No GPU adapter available in this environment; skip.
        };
        let gpu_result = adaptive_avg_pool2d(&gpu_input, (2, 2))
            .expect("test: GPU adaptive_avg_pool2d should succeed via CPU-delegate fallback");
        assert_eq!(gpu_result.shape().dims(), cpu_result.shape().dims());
        assert_eq!(
            gpu_result.to_vec().expect("test: to_vec should succeed"),
            cpu_result.to_vec().expect("test: to_vec should succeed"),
        );
    }

    #[test]
    fn gpu_adaptive_max_pool2d_matches_cpu_reference() {
        let data: Vec<f32> = (0..16).map(|v| v as f32).collect();
        let cpu_input =
            Tensor::<f32>::from_vec(data, &[1, 1, 4, 4]).expect("test: from_vec should succeed");
        let cpu_result = adaptive_max_pool2d(&cpu_input, (2, 2))
            .expect("test: CPU adaptive_max_pool2d should succeed");

        let gpu_input = match cpu_input.to(Device::Gpu(0)) {
            Ok(t) => t,
            Err(_) => return,
        };
        let gpu_result = adaptive_max_pool2d(&gpu_input, (2, 2))
            .expect("test: GPU adaptive_max_pool2d should succeed via CPU-delegate fallback");
        assert_eq!(gpu_result.shape().dims(), cpu_result.shape().dims());
        assert_eq!(
            gpu_result.to_vec().expect("test: to_vec should succeed"),
            cpu_result.to_vec().expect("test: to_vec should succeed"),
        );
    }
}
