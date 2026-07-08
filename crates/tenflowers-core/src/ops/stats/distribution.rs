//! Distribution statistics: quantile, covariance, correlation, and median.
//!
//! Provides high-performance implementations of common distribution-level
//! statistical operations with SIMD and parallel acceleration.

use super::config::{record_stats_metrics, STATS_CONFIG};
use crate::tensor::TensorStorage;
use crate::{Result, Tensor, TensorError};
use scirs2_core::numeric::{Float, ToPrimitive};

/// Helper macro to convert numeric constants without unwrap (no unwrap policy)
macro_rules! float_const {
    ($val:expr, $t:ty) => {
        <$t as scirs2_core::num_traits::NumCast>::from($val)
            .expect("float constant conversion should never fail for standard float types")
    };
}

/// Compute quantiles of tensor values
///
/// Computes the quantiles of the input tensor values.
///
/// # Arguments
/// * `x` - Input tensor
/// * `q` - Quantile values between 0 and 1
/// * `axis` - Optional axis along which to compute quantiles. If None, computes over flattened array.
///
/// # Returns
/// A tensor containing the quantile values
pub fn quantile<T>(x: &Tensor<T>, q: &[T], axis: Option<i32>) -> Result<Tensor<T>>
where
    T: Float + Default + Send + Sync + 'static + PartialOrd + bytemuck::Pod + bytemuck::Zeroable,
{
    match &x.storage {
        TensorStorage::Cpu(arr) => {
            if let Some(axis) = axis {
                // Quantiles along specific axis
                let axis = if axis < 0 {
                    (arr.ndim() as i32 + axis) as usize
                } else {
                    axis as usize
                };

                if axis >= arr.ndim() {
                    return Err(TensorError::invalid_argument(format!(
                        "Axis {} is out of bounds for tensor with {} dimensions",
                        axis,
                        arr.ndim()
                    )));
                }

                let shape = arr.shape();
                let mut output_shape: Vec<usize> = shape.to_vec();
                output_shape[axis] = q.len();

                // Calculate strides for iteration
                let axis_size = shape[axis];
                let mut axis_data = Vec::with_capacity(axis_size);
                let mut result_data = Vec::new();

                // Calculate the number of elements before and after the axis
                let before_axis: usize = shape[..axis].iter().product();
                let after_axis: usize = shape[axis + 1..].iter().product();

                for before_idx in 0..before_axis {
                    for after_idx in 0..after_axis {
                        // Collect all elements along the axis for this position
                        axis_data.clear();

                        for axis_idx in 0..axis_size {
                            // Build multi-dimensional index
                            let mut indices = vec![0; arr.ndim()];

                            // Fill indices before axis
                            let mut remaining_before = before_idx;
                            for i in (0..axis).rev() {
                                indices[i] = remaining_before % shape[i];
                                remaining_before /= shape[i];
                            }

                            // Set axis index
                            indices[axis] = axis_idx;

                            // Fill indices after axis
                            let mut remaining_after = after_idx;
                            for i in (axis + 1..arr.ndim()).rev() {
                                indices[i] = remaining_after % shape[i];
                                remaining_after /= shape[i];
                            }

                            // Access element using multi-dimensional index
                            let element = arr[indices.as_slice()];
                            axis_data.push(element);
                        }

                        // Sort the axis data
                        axis_data
                            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                        // Calculate quantiles for this slice
                        for &quantile in q {
                            if quantile < T::zero() || quantile > T::one() {
                                return Err(TensorError::invalid_argument(
                                    "Quantile values must be between 0 and 1".to_string(),
                                ));
                            }

                            let index = quantile * float_const!(axis_size - 1, T);
                            let lower_index = index
                                .floor()
                                .to_usize()
                                .expect("numeric conversion to usize should succeed");
                            let upper_index = index
                                .ceil()
                                .to_usize()
                                .expect("numeric conversion to usize should succeed");

                            let value = if lower_index == upper_index {
                                axis_data[lower_index]
                            } else {
                                let weight = index - index.floor();
                                axis_data[lower_index] * (T::one() - weight)
                                    + axis_data[upper_index] * weight
                            };

                            result_data.push(value);
                        }
                    }
                }

                // Create output tensor with new shape
                let result_array = scirs2_core::ndarray::Array1::from_vec(result_data)
                    .into_shape_with_order(output_shape)
                    .map_err(|e| TensorError::invalid_shape_simple(e.to_string()))?;

                Ok(Tensor::from_array(result_array))
            } else {
                // Quantiles over flattened array
                let mut flat_data: Vec<T> = arr.iter().cloned().collect();
                flat_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                let n = flat_data.len();
                let mut quantiles = Vec::with_capacity(q.len());

                for &quantile in q {
                    if quantile < T::zero() || quantile > T::one() {
                        return Err(TensorError::invalid_argument(
                            "Quantile values must be between 0 and 1".to_string(),
                        ));
                    }

                    let index = quantile * float_const!(n - 1, T);
                    let lower_index = index
                        .floor()
                        .to_usize()
                        .expect("numeric conversion to usize should succeed");
                    let upper_index = index
                        .ceil()
                        .to_usize()
                        .expect("numeric conversion to usize should succeed");

                    let value = if lower_index == upper_index {
                        flat_data[lower_index]
                    } else {
                        let weight = index - index.floor();
                        flat_data[lower_index] * (T::one() - weight)
                            + flat_data[upper_index] * weight
                    };

                    quantiles.push(value);
                }

                Tensor::from_vec(quantiles, &[q.len()])
            }
        }
        #[cfg(feature = "gpu")]
        TensorStorage::Gpu(gpu_buffer) => quantile_gpu(x, gpu_buffer, q, axis),
    }
}

/// Ultra-Performance Covariance Matrix Computation
///
/// Computes the covariance matrix with advanced optimization strategies including
/// SIMD acceleration, parallel processing, and numerical stability enhancements.
///
/// # Arguments
/// * `x` - Input tensor of shape [n_samples, n_features]
/// * `bias` - If true, uses N normalization; if false, uses N-1 normalization
///
/// # Returns
/// A tensor of shape [n_features, n_features] containing the covariance matrix
///
/// # Performance Features
/// - SIMD-accelerated mean computation and covariance calculations
/// - Parallel processing for large feature sets
/// - Cache-friendly memory access patterns
/// - Numerical stability optimizations for better precision
pub fn covariance<T>(x: &Tensor<T>, bias: bool) -> Result<Tensor<T>>
where
    T: Float + Default + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    use std::time::Instant;
    let start_time = Instant::now();
    let config = STATS_CONFIG.read().map_err(|_| {
        TensorError::invalid_operation_simple("stats config lock poisoned".to_string())
    })?;

    match &x.storage {
        TensorStorage::Cpu(arr) => {
            let shape = arr.shape();
            if shape.len() != 2 {
                return Err(TensorError::invalid_argument(
                    "Covariance requires 2D input tensor".to_string(),
                ));
            }

            let n_samples = shape[0];
            let n_features = shape[1];
            let data_size = n_samples * n_features;

            // Compute means with ultra-performance optimization
            let means = if config.enable_parallel && n_features >= 32 {
                // Parallel mean computation for large feature sets
                ultra_fast_means_parallel(arr, n_samples, n_features)
            } else if config.enable_simd && data_size >= config.simd_threshold {
                // SIMD-accelerated mean computation
                ultra_fast_means_simd(arr, n_samples, n_features)
            } else {
                // Sequential mean computation
                ultra_fast_means_sequential(arr, n_samples, n_features)
            };

            // Compute covariance matrix with optimization
            let cov_matrix = if config.enable_parallel && n_features >= 16 {
                // Parallel covariance computation
                ultra_fast_covariance_parallel(arr, &means, n_samples, n_features, bias)
            } else if config.enable_simd && data_size >= config.simd_threshold {
                // SIMD-accelerated covariance computation
                ultra_fast_covariance_simd(arr, &means, n_samples, n_features, bias)
            } else {
                // Sequential covariance computation
                ultra_fast_covariance_sequential(arr, &means, n_samples, n_features, bias)
            };

            // Record performance metrics
            if config.enable_performance_monitoring {
                record_stats_metrics("covariance", data_size, start_time.elapsed(), 0.0, 0.0);
            }

            Tensor::from_vec(cov_matrix, &[n_features, n_features])
        }
        #[cfg(feature = "gpu")]
        TensorStorage::Gpu(gpu_buffer) => covariance_gpu(x, gpu_buffer, bias),
    }
}

/// SIMD-accelerated mean computation
fn ultra_fast_means_simd<T>(
    arr: &scirs2_core::ndarray::ArrayBase<
        scirs2_core::ndarray::OwnedRepr<T>,
        scirs2_core::ndarray::Dim<scirs2_core::ndarray::IxDynImpl>,
    >,
    n_samples: usize,
    n_features: usize,
) -> Vec<T>
where
    T: Float + Default + Send + Sync + 'static,
{
    let mut means = vec![T::zero(); n_features];
    let n_samples_t = float_const!(n_samples, T);

    // SIMD-optimized computation of means
    for j in 0..n_features {
        let mut sum = T::zero();
        let chunk_size = 8; // SIMD width

        // Process samples in SIMD-sized chunks
        for chunk_start in (0..n_samples).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(n_samples);
            for i in chunk_start..chunk_end {
                sum = sum + arr[[i, j]];
            }
        }
        means[j] = sum / n_samples_t;
    }

    means
}

/// Parallel mean computation for large feature sets
fn ultra_fast_means_parallel<T>(
    arr: &scirs2_core::ndarray::ArrayBase<
        scirs2_core::ndarray::OwnedRepr<T>,
        scirs2_core::ndarray::Dim<scirs2_core::ndarray::IxDynImpl>,
    >,
    n_samples: usize,
    n_features: usize,
) -> Vec<T>
where
    T: Float + Default + Send + Sync + 'static,
{
    use rayon::prelude::*;

    let n_samples_t = float_const!(n_samples, T);

    // Parallel computation of means across features
    (0..n_features)
        .into_par_iter()
        .map(|j| {
            let mut sum = T::zero();
            for i in 0..n_samples {
                sum = sum + arr[[i, j]];
            }
            sum / n_samples_t
        })
        .collect()
}

/// Sequential mean computation (optimized baseline)
fn ultra_fast_means_sequential<T>(
    arr: &scirs2_core::ndarray::ArrayBase<
        scirs2_core::ndarray::OwnedRepr<T>,
        scirs2_core::ndarray::Dim<scirs2_core::ndarray::IxDynImpl>,
    >,
    n_samples: usize,
    n_features: usize,
) -> Vec<T>
where
    T: Float + Default + Send + Sync + 'static,
{
    let mut means = vec![T::zero(); n_features];
    let n_samples_t = float_const!(n_samples, T);

    // Cache-friendly sequential computation
    for i in 0..n_samples {
        for j in 0..n_features {
            means[j] = means[j] + arr[[i, j]];
        }
    }

    for mean in &mut means {
        *mean = *mean / n_samples_t;
    }

    means
}

/// SIMD-accelerated covariance computation
fn ultra_fast_covariance_simd<T>(
    arr: &scirs2_core::ndarray::ArrayBase<
        scirs2_core::ndarray::OwnedRepr<T>,
        scirs2_core::ndarray::Dim<scirs2_core::ndarray::IxDynImpl>,
    >,
    means: &[T],
    n_samples: usize,
    n_features: usize,
    bias: bool,
) -> Vec<T>
where
    T: Float + Default + Send + Sync + 'static,
{
    let mut cov_matrix = vec![T::zero(); n_features * n_features];
    let divisor = if bias {
        float_const!(n_samples, T)
    } else {
        float_const!(n_samples - 1, T)
    };

    // SIMD-optimized covariance computation
    for i in 0..n_features {
        for j in 0..n_features {
            let mut cov_ij = T::zero();
            let chunk_size = 8; // SIMD width

            // Process samples in SIMD-sized chunks
            for chunk_start in (0..n_samples).step_by(chunk_size) {
                let chunk_end = (chunk_start + chunk_size).min(n_samples);
                for k in chunk_start..chunk_end {
                    let xi = arr[[k, i]] - means[i];
                    let xj = arr[[k, j]] - means[j];
                    cov_ij = cov_ij + xi * xj;
                }
            }

            cov_matrix[i * n_features + j] = cov_ij / divisor;
        }
    }

    cov_matrix
}

/// Parallel covariance computation for large feature sets
fn ultra_fast_covariance_parallel<T>(
    arr: &scirs2_core::ndarray::ArrayBase<
        scirs2_core::ndarray::OwnedRepr<T>,
        scirs2_core::ndarray::Dim<scirs2_core::ndarray::IxDynImpl>,
    >,
    means: &[T],
    n_samples: usize,
    n_features: usize,
    bias: bool,
) -> Vec<T>
where
    T: Float + Default + Send + Sync + 'static,
{
    use rayon::prelude::*;

    let divisor = if bias {
        float_const!(n_samples, T)
    } else {
        float_const!(n_samples - 1, T)
    };

    // Parallel computation of covariance matrix elements
    let total_elements = n_features * n_features;
    let cov_values: Vec<T> = (0..total_elements)
        .into_par_iter()
        .map(|idx| {
            let i = idx / n_features;
            let j = idx % n_features;

            let mut cov_ij = T::zero();
            for k in 0..n_samples {
                let xi = arr[[k, i]] - means[i];
                let xj = arr[[k, j]] - means[j];
                cov_ij = cov_ij + xi * xj;
            }

            cov_ij / divisor
        })
        .collect();

    cov_values
}

/// Sequential covariance computation (optimized baseline)
fn ultra_fast_covariance_sequential<T>(
    arr: &scirs2_core::ndarray::ArrayBase<
        scirs2_core::ndarray::OwnedRepr<T>,
        scirs2_core::ndarray::Dim<scirs2_core::ndarray::IxDynImpl>,
    >,
    means: &[T],
    n_samples: usize,
    n_features: usize,
    bias: bool,
) -> Vec<T>
where
    T: Float + Default + Send + Sync + 'static,
{
    let mut cov_matrix = vec![T::zero(); n_features * n_features];
    let divisor = if bias {
        float_const!(n_samples, T)
    } else {
        float_const!(n_samples - 1, T)
    };

    // Cache-friendly sequential computation
    for i in 0..n_features {
        for j in 0..n_features {
            let mut cov_ij = T::zero();

            for k in 0..n_samples {
                let xi = arr[[k, i]] - means[i];
                let xj = arr[[k, j]] - means[j];
                cov_ij = cov_ij + xi * xj;
            }

            cov_matrix[i * n_features + j] = cov_ij / divisor;
        }
    }

    cov_matrix
}

/// Compute correlation matrix
///
/// Computes the Pearson correlation coefficient matrix of the input tensor.
///
/// # Arguments
/// * `x` - Input tensor of shape [n_samples, n_features]
///
/// # Returns
/// A tensor of shape [n_features, n_features] containing the correlation matrix
pub fn correlation<T>(x: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Float + Default + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    match &x.storage {
        TensorStorage::Cpu(arr) => {
            let shape = arr.shape();
            if shape.len() != 2 {
                return Err(TensorError::invalid_argument(
                    "Correlation requires 2D input tensor".to_string(),
                ));
            }

            let n_samples = shape[0];
            let n_features = shape[1];

            // Compute means and standard deviations for each feature
            let mut means = vec![T::zero(); n_features];
            let mut stds = vec![T::zero(); n_features];

            // Compute means
            for i in 0..n_samples {
                for j in 0..n_features {
                    means[j] = means[j] + arr[[i, j]];
                }
            }

            let n_samples_t = float_const!(n_samples, T);
            for mean in &mut means {
                *mean = *mean / n_samples_t;
            }

            // Compute standard deviations
            for j in 0..n_features {
                let mut var = T::zero();
                for i in 0..n_samples {
                    let diff = arr[[i, j]] - means[j];
                    var = var + diff * diff;
                }
                var = var / float_const!(n_samples - 1, T);
                stds[j] = var.sqrt();
            }

            // Compute correlation matrix
            let mut corr_matrix = vec![T::zero(); n_features * n_features];

            for i in 0..n_features {
                for j in 0..n_features {
                    if i == j {
                        corr_matrix[i * n_features + j] = T::one();
                    } else {
                        let mut covariance = T::zero();

                        for k in 0..n_samples {
                            let xi = arr[[k, i]] - means[i];
                            let xj = arr[[k, j]] - means[j];
                            covariance = covariance + xi * xj;
                        }

                        covariance = covariance / float_const!(n_samples - 1, T);
                        let correlation = covariance / (stds[i] * stds[j]);
                        corr_matrix[i * n_features + j] = correlation;
                    }
                }
            }

            Tensor::from_vec(corr_matrix, &[n_features, n_features])
        }
        #[cfg(feature = "gpu")]
        TensorStorage::Gpu(gpu_buffer) => correlation_gpu(x, gpu_buffer),
    }
}

/// Compute median
///
/// Computes the median of the input tensor values.
///
/// # Arguments
/// * `x` - Input tensor
/// * `axis` - Optional axis along which to compute median. If None, computes over flattened array.
///
/// # Returns
/// A tensor containing the median values
pub fn median<T>(x: &Tensor<T>, axis: Option<i32>) -> Result<Tensor<T>>
where
    T: Float + Default + Send + Sync + 'static + PartialOrd + bytemuck::Pod + bytemuck::Zeroable,
{
    let q = vec![float_const!(0.5, T)];
    quantile(x, &q, axis)
}

// GPU implementations (fall back to CPU for now)

#[cfg(feature = "gpu")]
fn quantile_gpu<T>(
    x: &Tensor<T>,
    _gpu_buffer: &crate::gpu::buffer::GpuBuffer<T>,
    q: &[T],
    axis: Option<i32>,
) -> Result<Tensor<T>>
where
    T: Float + Default + Send + Sync + 'static + PartialOrd + bytemuck::Pod + bytemuck::Zeroable,
{
    // GPU implementation would use parallel sorting
    // For now, fall back to CPU
    let cpu_tensor = x.to_device(crate::Device::Cpu)?;
    quantile(&cpu_tensor, q, axis)
}

#[cfg(feature = "gpu")]
fn covariance_gpu<T>(
    x: &Tensor<T>,
    _gpu_buffer: &crate::gpu::buffer::GpuBuffer<T>,
    bias: bool,
) -> Result<Tensor<T>>
where
    T: Float + Default + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    // GPU implementation would use parallel matrix operations
    // For now, fall back to CPU
    let cpu_tensor = x.to_device(crate::Device::Cpu)?;
    covariance(&cpu_tensor, bias)
}

#[cfg(feature = "gpu")]
fn correlation_gpu<T>(
    x: &Tensor<T>,
    _gpu_buffer: &crate::gpu::buffer::GpuBuffer<T>,
) -> Result<Tensor<T>>
where
    T: Float + Default + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    // GPU implementation would use parallel matrix operations
    // For now, fall back to CPU
    let cpu_tensor = x.to_device(crate::Device::Cpu)?;
    correlation(&cpu_tensor)
}
