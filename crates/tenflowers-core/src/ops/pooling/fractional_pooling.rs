//! Fractional pooling operations for 2D tensors
//!
//! Fractional pooling uses stochastic or deterministic sampling to achieve
//! flexible downsampling ratios that aren't constrained to integer factors.

use crate::tensor::TensorStorage;
use crate::{Result, Tensor, TensorError};
use scirs2_core::numeric::{Float, FromPrimitive, Zero};
use scirs2_core::random::{Random, Rng};

/// Fractional max pooling 2D operation
/// Uses stochastic or deterministic fractional scaling
pub fn fractional_max_pool2d<T>(
    input: &Tensor<T>,
    pooling_ratio: (f32, f32),
    random_samples: Option<&Tensor<T>>,
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + PartialOrd
        + Float
        + FromPrimitive
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    match &input.storage {
        TensorStorage::Cpu(_input_arr) => {
            fractional_max_pool2d_cpu(input, pooling_ratio, random_samples)
        }
        #[cfg(feature = "gpu")]
        TensorStorage::Gpu(_gpu_buffer) => {
            fractional_max_pool2d_gpu(input, pooling_ratio, random_samples)
        }
    }
}

/// Fractional average pooling 2D operation
/// Uses stochastic or deterministic fractional scaling
pub fn fractional_avg_pool2d<T>(
    input: &Tensor<T>,
    pooling_ratio: (f32, f32),
    random_samples: Option<&Tensor<T>>,
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
        TensorStorage::Cpu(_input_arr) => {
            fractional_avg_pool2d_cpu(input, pooling_ratio, random_samples)
        }
        #[cfg(feature = "gpu")]
        TensorStorage::Gpu(_gpu_buffer) => {
            fractional_avg_pool2d_gpu(input, pooling_ratio, random_samples)
        }
    }
}

// CPU implementations

fn fractional_max_pool2d_cpu<T>(
    input: &Tensor<T>,
    pooling_ratio: (f32, f32),
    random_samples: Option<&Tensor<T>>,
) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + PartialOrd + Float + FromPrimitive + Send + Sync + 'static,
{
    let shape = input.shape();
    if shape.rank() != 4 {
        return Err(TensorError::invalid_shape_simple(format!(
            "FractionalMaxPool2D expects 4D input, got {}D",
            shape.rank()
        )));
    }

    let batch_size = shape.dims()[0];
    let channels = shape.dims()[1];
    let input_height = shape.dims()[2];
    let input_width = shape.dims()[3];

    let output_height = (input_height as f32 * pooling_ratio.0) as usize;
    let output_width = (input_width as f32 * pooling_ratio.1) as usize;

    // Generate pooling regions
    let (row_splits, col_splits) = if let Some(samples) = random_samples {
        // Use provided random samples for deterministic behavior
        generate_pooling_regions_deterministic(
            input_height,
            input_width,
            output_height,
            output_width,
            samples,
        )?
    } else {
        // Generate random pooling regions
        generate_pooling_regions_random(input_height, input_width, output_height, output_width)?
    };

    let mut output_data = vec![T::zero(); batch_size * channels * output_height * output_width];

    for b in 0..batch_size {
        for c in 0..channels {
            for oh in 0..output_height {
                for ow in 0..output_width {
                    let h_start = row_splits[oh];
                    let h_end = row_splits[oh + 1];
                    let w_start = col_splits[ow];
                    let w_end = col_splits[ow + 1];

                    let mut max_val: Option<T> = None;
                    for h in h_start..h_end {
                        for w in w_start..w_end {
                            if let Some(val) = input.get(&[b, c, h, w]) {
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

                    let out_idx = b * channels * output_height * output_width
                        + c * output_height * output_width
                        + oh * output_width
                        + ow;
                    output_data[out_idx] = max_val.unwrap_or_else(T::zero);
                }
            }
        }
    }

    Tensor::from_vec(
        output_data,
        &[batch_size, channels, output_height, output_width],
    )
}

fn fractional_avg_pool2d_cpu<T>(
    input: &Tensor<T>,
    pooling_ratio: (f32, f32),
    random_samples: Option<&Tensor<T>>,
) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + Float + FromPrimitive + Send + Sync + 'static,
{
    let shape = input.shape();
    if shape.rank() != 4 {
        return Err(TensorError::invalid_shape_simple(format!(
            "FractionalAvgPool2D expects 4D input, got {}D",
            shape.rank()
        )));
    }

    let batch_size = shape.dims()[0];
    let channels = shape.dims()[1];
    let input_height = shape.dims()[2];
    let input_width = shape.dims()[3];

    let output_height = (input_height as f32 * pooling_ratio.0) as usize;
    let output_width = (input_width as f32 * pooling_ratio.1) as usize;

    // Generate pooling regions
    let (row_splits, col_splits) = if let Some(samples) = random_samples {
        // Use provided random samples for deterministic behavior
        generate_pooling_regions_deterministic(
            input_height,
            input_width,
            output_height,
            output_width,
            samples,
        )?
    } else {
        // Generate random pooling regions
        generate_pooling_regions_random(input_height, input_width, output_height, output_width)?
    };

    let mut output_data = vec![T::zero(); batch_size * channels * output_height * output_width];

    for b in 0..batch_size {
        for c in 0..channels {
            for oh in 0..output_height {
                for ow in 0..output_width {
                    let h_start = row_splits[oh];
                    let h_end = row_splits[oh + 1];
                    let w_start = col_splits[ow];
                    let w_end = col_splits[ow + 1];

                    let mut sum = T::zero();
                    let mut count = 0;

                    for h in h_start..h_end {
                        for w in w_start..w_end {
                            if let Some(val) = input.get(&[b, c, h, w]) {
                                sum = sum + val;
                                count += 1;
                            }
                        }
                    }

                    let out_idx = b * channels * output_height * output_width
                        + c * output_height * output_width
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

    Tensor::from_vec(
        output_data,
        &[batch_size, channels, output_height, output_width],
    )
}

// Helper functions for generating pooling regions

/// Generate random pooling regions for fractional pooling
fn generate_pooling_regions_random(
    input_height: usize,
    input_width: usize,
    output_height: usize,
    output_width: usize,
) -> Result<(Vec<usize>, Vec<usize>)> {
    let mut rng = scirs2_core::random::rng();

    // Generate random split points for rows
    let mut row_splits = vec![0];
    for _ in 0..output_height {
        let last_split = *row_splits.last().unwrap();
        let remaining_height = input_height - last_split;
        let remaining_outputs = output_height - (row_splits.len() - 1);

        if remaining_outputs == 0 {
            break;
        }

        let min_step = remaining_height / remaining_outputs;
        let max_step = if remaining_outputs == 1 {
            remaining_height
        } else {
            std::cmp::min(remaining_height, min_step * 2)
        };

        let step = if min_step == max_step {
            min_step
        } else {
            rng.random_range(min_step..=max_step)
        };

        row_splits.push(last_split + step);
    }
    row_splits.push(input_height);

    // Generate random split points for columns
    let mut col_splits = vec![0];
    for _ in 0..output_width {
        let last_split = *col_splits.last().unwrap();
        let remaining_width = input_width - last_split;
        let remaining_outputs = output_width - (col_splits.len() - 1);

        if remaining_outputs == 0 {
            break;
        }

        let min_step = remaining_width / remaining_outputs;
        let max_step = if remaining_outputs == 1 {
            remaining_width
        } else {
            std::cmp::min(remaining_width, min_step * 2)
        };

        let step = if min_step == max_step {
            min_step
        } else {
            rng.random_range(min_step..=max_step)
        };

        col_splits.push(last_split + step);
    }
    col_splits.push(input_width);

    Ok((row_splits, col_splits))
}

/// Generate deterministic pooling regions for fractional pooling
fn generate_pooling_regions_deterministic<T>(
    input_height: usize,
    input_width: usize,
    output_height: usize,
    output_width: usize,
    random_samples: &Tensor<T>,
) -> Result<(Vec<usize>, Vec<usize>)>
where
    T: Clone + Float + FromPrimitive,
{
    // For deterministic fractional pooling, we use the provided random samples
    // to generate consistent pooling regions
    let samples = random_samples.as_slice().ok_or_else(|| {
        TensorError::device_error_simple("Cannot access random samples tensor data".to_string())
    })?;

    let expected_samples = output_height + output_width;
    if samples.len() < expected_samples {
        return Err(TensorError::invalid_shape_simple(format!(
            "Need at least {} random samples, got {}",
            expected_samples,
            samples.len()
        )));
    }

    // Generate deterministic split points for rows
    let mut row_splits = vec![0];
    #[allow(clippy::needless_range_loop)]
    for i in 0..output_height {
        let last_split = *row_splits.last().unwrap();
        let remaining_height = input_height - last_split;
        let remaining_outputs = output_height - (row_splits.len() - 1);

        if remaining_outputs == 0 {
            break;
        }

        let min_step = remaining_height / remaining_outputs;
        let max_step = if remaining_outputs == 1 {
            remaining_height
        } else {
            std::cmp::min(remaining_height, min_step * 2)
        };

        let random_val = samples[i].to_f32().unwrap_or(0.5);
        let step = min_step + ((max_step - min_step) as f32 * random_val) as usize;
        row_splits.push(last_split + step);
    }
    row_splits.push(input_height);

    // Generate deterministic split points for columns
    let mut col_splits = vec![0];
    #[allow(clippy::needless_range_loop)]
    for i in 0..output_width {
        let last_split = *col_splits.last().unwrap();
        let remaining_width = input_width - last_split;
        let remaining_outputs = output_width - (col_splits.len() - 1);

        if remaining_outputs == 0 {
            break;
        }

        let min_step = remaining_width / remaining_outputs;
        let max_step = if remaining_outputs == 1 {
            remaining_width
        } else {
            std::cmp::min(remaining_width, min_step * 2)
        };

        let random_val = samples[output_height + i].to_f32().unwrap_or(0.5);
        let step = min_step + ((max_step - min_step) as f32 * random_val) as usize;
        col_splits.push(last_split + step);
    }
    col_splits.push(input_width);

    Ok((row_splits, col_splits))
}

// GPU implementations

#[cfg(feature = "gpu")]
fn fractional_max_pool2d_gpu<T>(
    input: &Tensor<T>,
    pooling_ratio: (f32, f32),
    random_samples: Option<&Tensor<T>>,
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + PartialOrd
        + Float
        + FromPrimitive
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    if let TensorStorage::Gpu(gpu_buffer) = &input.storage {
        let input_shape = input.shape().dims();
        if input_shape.len() != 4 {
            return Err(TensorError::invalid_shape_simple(
                "Fractional max pool input must be 4D (NCHW format)".to_string(),
            ));
        }

        let batch_size = input_shape[0];
        let channels = input_shape[1];
        let input_height = input_shape[2];
        let input_width = input_shape[3];

        // Calculate output dimensions based on pooling ratio
        let output_height = (input_height as f32 * pooling_ratio.0).round() as usize;
        let output_width = (input_width as f32 * pooling_ratio.1).round() as usize;
        let output_shape = vec![batch_size, channels, output_height, output_width];

        let pooling_ratio_slice = &[pooling_ratio.0, pooling_ratio.1];
        let output_len = output_shape.iter().product();

        let result_gpu = crate::gpu::ops::execute_fractional_pooling_op(
            gpu_buffer,
            crate::gpu::ops::PoolingOp::FractionalMaxPool2D,
            pooling_ratio_slice,
            false, // pseudo_random
            false, // overlapping
            input_shape,
            output_len,
        )?;

        let mut result = Tensor::from_gpu_buffer(result_gpu, crate::Shape::new(output_shape));
        result.set_requires_grad(input.requires_grad());
        Ok(result)
    } else {
        // Fallback to CPU implementation for non-GPU tensors
        fractional_max_pool2d_cpu(input, pooling_ratio, random_samples)
    }
}

#[cfg(feature = "gpu")]
fn fractional_avg_pool2d_gpu<T>(
    input: &Tensor<T>,
    pooling_ratio: (f32, f32),
    random_samples: Option<&Tensor<T>>,
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
    if let TensorStorage::Gpu(gpu_buffer) = &input.storage {
        let input_shape = input.shape().dims();
        if input_shape.len() != 4 {
            return Err(TensorError::invalid_shape_simple(
                "Fractional avg pool input must be 4D (NCHW format)".to_string(),
            ));
        }

        let batch_size = input_shape[0];
        let channels = input_shape[1];
        let input_height = input_shape[2];
        let input_width = input_shape[3];

        // Calculate output dimensions based on pooling ratio
        let output_height = (input_height as f32 * pooling_ratio.0).round() as usize;
        let output_width = (input_width as f32 * pooling_ratio.1).round() as usize;
        let output_shape = vec![batch_size, channels, output_height, output_width];

        let pooling_ratio_slice = &[pooling_ratio.0, pooling_ratio.1];
        let output_len = output_shape.iter().product();

        let result_gpu = crate::gpu::ops::execute_fractional_pooling_op(
            gpu_buffer,
            crate::gpu::ops::PoolingOp::FractionalAvgPool2D,
            pooling_ratio_slice,
            false, // pseudo_random
            false, // overlapping
            input_shape,
            output_len,
        )?;

        let mut result = Tensor::from_gpu_buffer(result_gpu, crate::Shape::new(output_shape));
        result.set_requires_grad(input.requires_grad());
        Ok(result)
    } else {
        // Fallback to CPU implementation for non-GPU tensors
        fractional_avg_pool2d_cpu(input, pooling_ratio, random_samples)
    }
}
