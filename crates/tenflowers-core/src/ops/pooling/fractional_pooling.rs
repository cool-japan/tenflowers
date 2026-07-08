//! Fractional pooling operations for 2D tensors
//!
//! Fractional pooling uses stochastic or deterministic sampling to achieve
//! flexible downsampling ratios that aren't constrained to integer factors.

use crate::tensor::TensorStorage;
use crate::{Result, Tensor, TensorError};
use scirs2_core::numeric::{Float, FromPrimitive, Zero};
use scirs2_core::random::{Random, Rng};
use scirs2_core::RngExt;

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
                        output_data[out_idx] = sum
                            / T::from(count).expect("count must be convertible to tensor dtype");
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
        let last_split = *row_splits.last().expect("collection should not be empty");
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
        let last_split = *col_splits.last().expect("collection should not be empty");
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
        let last_split = *row_splits.last().expect("collection should not be empty");
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
        let last_split = *col_splits.last().expect("collection should not be empty");
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

// `execute_fractional_pooling_op`'s `shader_entry_point` dispatch (in
// `crate::gpu::ops::pooling_ops`) only has real kernels for
// `PoolingOp::{MaxPool2D, GlobalMaxPool, AvgPool2D, GlobalAvgPool}`; passing
// `PoolingOp::FractionalMaxPool2D` / `FractionalAvgPool2D` would always fall
// through to its wildcard arm and return an honest error. Rather than call a
// non-existent kernel, read the operand(s) back to the host (a real
// device->host transfer, or a no-op clone if already CPU-resident) and
// delegate to the CPU implementation, which is known-correct. Both `input`
// AND `random_samples` (when present) must be brought to the host: the CPU
// implementation's `generate_pooling_regions_deterministic` reads
// `random_samples` via `.as_slice()`, which returns `None` for GPU-resident
// storage.
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
    let cpu_input = input.to_cpu()?;
    let cpu_samples = random_samples.map(|s| s.to_cpu()).transpose()?;
    let result = fractional_max_pool2d_cpu(&cpu_input, pooling_ratio, cpu_samples.as_ref())?;
    result.to_device(input.device().clone())
}

// See the comment on `fractional_max_pool2d_gpu` above: `PoolingOp::
// FractionalAvgPool2D` never matches a real kernel in `execute_fractional_
// pooling_op`'s shader dispatch either, so this delegates to the CPU
// implementation after bringing both `input` and `random_samples` (when
// present) to the host.
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
    let cpu_input = input.to_cpu()?;
    let cpu_samples = random_samples.map(|s| s.to_cpu()).transpose()?;
    let result = fractional_avg_pool2d_cpu(&cpu_input, pooling_ratio, cpu_samples.as_ref())?;
    result.to_device(input.device().clone())
}

// GPU delegate correctness tests: verify fractional_max_pool2d_gpu/
// fractional_avg_pool2d_gpu round-trip GPU-resident input (and, when
// present, a GPU-resident `random_samples` tensor) through the host and
// delegate to the known-correct CPU implementation, instead of erroring via
// execute_fractional_pooling_op (whose FractionalMaxPool2D/FractionalAvgPool2D
// op values never match any of that dispatcher's shader_entry_point arms).
//
// These also regression-test a separate bug: the old GPU wrapper accepted
// `random_samples: Option<&Tensor<T>>` in its signature but never forwarded
// it to the CPU implementation, so a GPU-resident samples tensor would fail
// `generate_pooling_regions_deterministic`'s `.as_slice()` call. The fix
// converts `random_samples` to CPU too, alongside the primary input.
#[cfg(all(test, feature = "gpu"))]
mod gpu_delegate_tests {
    use super::*;
    use crate::Device;

    #[test]
    fn gpu_fractional_max_pool2d_with_random_samples_matches_cpu_reference() {
        let input_data: Vec<f32> = (0..16).map(|v| v as f32).collect();
        let cpu_input = Tensor::<f32>::from_vec(input_data, &[1, 1, 4, 4])
            .expect("test: from_vec should succeed");
        let samples_data = vec![0.25f32, 0.75, 0.1, 0.9]; // output_height(2) + output_width(2)
        let cpu_samples =
            Tensor::<f32>::from_vec(samples_data, &[4]).expect("test: from_vec should succeed");

        let cpu_result = fractional_max_pool2d(&cpu_input, (0.5, 0.5), Some(&cpu_samples))
            .expect("test: CPU fractional_max_pool2d should succeed");

        let (gpu_input, gpu_samples) =
            match (cpu_input.to(Device::Gpu(0)), cpu_samples.to(Device::Gpu(0))) {
                (Ok(i), Ok(s)) => (i, s),
                _ => return, // No GPU adapter available in this environment; skip.
            };

        let gpu_result = fractional_max_pool2d(&gpu_input, (0.5, 0.5), Some(&gpu_samples)).expect(
            "test: GPU fractional_max_pool2d with random_samples must succeed \
             (regression test: random_samples must not be silently dropped)",
        );
        assert_eq!(gpu_result.shape().dims(), cpu_result.shape().dims());
        assert_eq!(
            gpu_result.to_vec().expect("test: to_vec should succeed"),
            cpu_result.to_vec().expect("test: to_vec should succeed"),
        );
    }

    #[test]
    fn gpu_fractional_avg_pool2d_with_random_samples_matches_cpu_reference() {
        let input_data: Vec<f32> = (0..16).map(|v| v as f32).collect();
        let cpu_input = Tensor::<f32>::from_vec(input_data, &[1, 1, 4, 4])
            .expect("test: from_vec should succeed");
        let samples_data = vec![0.25f32, 0.75, 0.1, 0.9];
        let cpu_samples =
            Tensor::<f32>::from_vec(samples_data, &[4]).expect("test: from_vec should succeed");

        let cpu_result = fractional_avg_pool2d(&cpu_input, (0.5, 0.5), Some(&cpu_samples))
            .expect("test: CPU fractional_avg_pool2d should succeed");

        let (gpu_input, gpu_samples) =
            match (cpu_input.to(Device::Gpu(0)), cpu_samples.to(Device::Gpu(0))) {
                (Ok(i), Ok(s)) => (i, s),
                _ => return,
            };

        let gpu_result = fractional_avg_pool2d(&gpu_input, (0.5, 0.5), Some(&gpu_samples)).expect(
            "test: GPU fractional_avg_pool2d with random_samples must succeed \
             (regression test: random_samples must not be silently dropped)",
        );
        assert_eq!(gpu_result.shape().dims(), cpu_result.shape().dims());
        assert_eq!(
            gpu_result.to_vec().expect("test: to_vec should succeed"),
            cpu_result.to_vec().expect("test: to_vec should succeed"),
        );
    }

    #[test]
    fn gpu_fractional_max_pool2d_without_samples_succeeds() {
        let input_data: Vec<f32> = (0..16).map(|v| v as f32).collect();
        let cpu_input = Tensor::<f32>::from_vec(input_data, &[1, 1, 4, 4])
            .expect("test: from_vec should succeed");
        let gpu_input = match cpu_input.to(Device::Gpu(0)) {
            Ok(t) => t,
            Err(_) => return,
        };
        let gpu_result = fractional_max_pool2d(&gpu_input, (0.5, 0.5), None).expect(
            "test: GPU fractional_max_pool2d without samples should succeed via CPU-delegate fallback",
        );
        assert_eq!(gpu_result.shape().dims(), &[1, 1, 2, 2]);
    }

    #[test]
    fn gpu_fractional_avg_pool2d_without_samples_succeeds() {
        let input_data: Vec<f32> = (0..16).map(|v| v as f32).collect();
        let cpu_input = Tensor::<f32>::from_vec(input_data, &[1, 1, 4, 4])
            .expect("test: from_vec should succeed");
        let gpu_input = match cpu_input.to(Device::Gpu(0)) {
            Ok(t) => t,
            Err(_) => return,
        };
        let gpu_result = fractional_avg_pool2d(&gpu_input, (0.5, 0.5), None).expect(
            "test: GPU fractional_avg_pool2d without samples should succeed via CPU-delegate fallback",
        );
        assert_eq!(gpu_result.shape().dims(), &[1, 1, 2, 2]);
    }
}
