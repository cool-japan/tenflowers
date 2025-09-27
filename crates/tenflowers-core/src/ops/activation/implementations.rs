use crate::{Result, TensorError};
use num_traits::{Float, Zero};
use rayon::prelude::*;
use scirs2_autograd::ndarray::ArrayD;
use std::time::Instant;

use super::core::{get_activation_registry, ActivationStrategy};
use super::simd;
use super::strategy::select_activation_strategy;

/// Ultra-performance ReLU implementation with adaptive optimization
pub fn ultra_relu_vectorized<T>(arr: &ArrayD<T>) -> Result<ArrayD<T>>
where
    T: Copy + Default + Zero + PartialOrd + Send + Sync + 'static,
{
    let registry = get_activation_registry();
    let start_time = Instant::now();
    let total_elements = arr.len();
    let strategy = select_activation_strategy(total_elements, false); // ReLU is not transcendental

    let result = match strategy {
        ActivationStrategy::Sequential => {
            let zero = T::zero();
            arr.mapv(|v| if v > zero { v } else { zero })
        }
        ActivationStrategy::Simd => {
            // Special handling for f32 SIMD
            if std::any::type_name::<T>() == "f32" && arr.is_standard_layout() {
                if let Some(input_slice) = arr.as_slice() {
                    let input_f32 = unsafe {
                        std::slice::from_raw_parts(
                            input_slice.as_ptr() as *const f32,
                            input_slice.len(),
                        )
                    };
                    let mut output_data = vec![0.0f32; input_f32.len()];

                    match simd::simd_relu_f32(input_f32, &mut output_data) {
                        Ok(()) => {
                            registry.record_simd();
                            let output_t = unsafe {
                                std::slice::from_raw_parts(
                                    output_data.as_ptr() as *const T,
                                    output_data.len(),
                                )
                            };
                            ArrayD::from_shape_vec(arr.raw_dim(), output_t.to_vec()).map_err(
                                |e| {
                                    TensorError::invalid_argument(format!(
                                        "SIMD ReLU shape error: {}",
                                        e
                                    ))
                                },
                            )?
                        }
                        Err(_) => {
                            // Fallback to sequential
                            let zero = T::zero();
                            arr.mapv(|v| if v > zero { v } else { zero })
                        }
                    }
                } else {
                    let zero = T::zero();
                    arr.mapv(|v| if v > zero { v } else { zero })
                }
            } else {
                let zero = T::zero();
                arr.mapv(|v| if v > zero { v } else { zero })
            }
        }
        ActivationStrategy::Parallel => {
            registry.record_parallel();
            let zero = T::zero();
            if let Some(data_slice) = arr.as_slice() {
                let mut result = ArrayD::zeros(arr.raw_dim());
                if let Some(result_slice) = result.as_slice_mut() {
                    result_slice
                        .par_iter_mut()
                        .zip(data_slice.par_iter())
                        .for_each(|(out, &input)| {
                            *out = if input > zero { input } else { zero };
                        });
                    result
                } else {
                    arr.mapv(|v| if v > zero { v } else { zero })
                }
            } else {
                arr.mapv(|v| if v > zero { v } else { zero })
            }
        }
        ActivationStrategy::SimdParallel => {
            registry.record_simd();
            registry.record_parallel();
            // For now, use parallel implementation with potential SIMD in chunks
            let zero = T::zero();
            if let Some(data_slice) = arr.as_slice() {
                let mut result = ArrayD::zeros(arr.raw_dim());
                if let Some(result_slice) = result.as_slice_mut() {
                    const CHUNK_SIZE: usize = 8192;
                    result_slice
                        .par_chunks_mut(CHUNK_SIZE)
                        .zip(data_slice.par_chunks(CHUNK_SIZE))
                        .for_each(|(output_chunk, input_chunk)| {
                            for (out, &inp) in output_chunk.iter_mut().zip(input_chunk.iter()) {
                                *out = if inp > zero { inp } else { zero };
                            }
                        });
                    result
                } else {
                    arr.mapv(|v| if v > zero { v } else { zero })
                }
            } else {
                arr.mapv(|v| if v > zero { v } else { zero })
            }
        }
        _ => {
            let zero = T::zero();
            arr.mapv(|v| if v > zero { v } else { zero })
        }
    };

    // Record performance metrics
    let duration = start_time.elapsed();
    registry.record_function("relu", total_elements, duration.as_nanos() as u64);

    Ok(result)
}

/// Ultra-performance sigmoid implementation with adaptive optimization
pub fn ultra_sigmoid_vectorized<T>(arr: &ArrayD<T>) -> Result<ArrayD<T>>
where
    T: Float + Send + Sync + 'static,
{
    let registry = get_activation_registry();
    let start_time = Instant::now();
    let total_elements = arr.len();
    let strategy = select_activation_strategy(total_elements, true); // Sigmoid is transcendental

    let result = match strategy {
        ActivationStrategy::Simd => {
            // Special handling for f32 SIMD
            if std::any::type_name::<T>() == "f32" && arr.is_standard_layout() {
                if let Some(input_slice) = arr.as_slice() {
                    let input_f32 = unsafe {
                        std::slice::from_raw_parts(
                            input_slice.as_ptr() as *const f32,
                            input_slice.len(),
                        )
                    };
                    let mut output_data = vec![0.0f32; input_f32.len()];

                    match simd::simd_sigmoid_f32(input_f32, &mut output_data) {
                        Ok(()) => {
                            registry.record_simd();
                            let output_t = unsafe {
                                std::slice::from_raw_parts(
                                    output_data.as_ptr() as *const T,
                                    output_data.len(),
                                )
                            };
                            ArrayD::from_shape_vec(arr.raw_dim(), output_t.to_vec()).map_err(
                                |e| {
                                    TensorError::invalid_argument(format!(
                                        "SIMD sigmoid shape error: {}",
                                        e
                                    ))
                                },
                            )?
                        }
                        Err(_) => {
                            // Fallback to standard implementation
                            arr.mapv(|v| T::one() / (T::one() + (-v).exp()))
                        }
                    }
                } else {
                    arr.mapv(|v| T::one() / (T::one() + (-v).exp()))
                }
            } else {
                arr.mapv(|v| T::one() / (T::one() + (-v).exp()))
            }
        }
        ActivationStrategy::Parallel => {
            registry.record_parallel();
            let one = T::one();
            if let Some(data_slice) = arr.as_slice() {
                let mut result = ArrayD::zeros(arr.raw_dim());
                if let Some(result_slice) = result.as_slice_mut() {
                    result_slice
                        .par_iter_mut()
                        .zip(data_slice.par_iter())
                        .for_each(|(out, &input)| {
                            *out = one / (one + (-input).exp());
                        });
                    result
                } else {
                    arr.mapv(|v| one / (one + (-v).exp()))
                }
            } else {
                arr.mapv(|v| one / (one + (-v).exp()))
            }
        }
        ActivationStrategy::SimdParallel => {
            registry.record_simd();
            registry.record_parallel();
            let one = T::one();
            if let Some(data_slice) = arr.as_slice() {
                let mut result = ArrayD::zeros(arr.raw_dim());
                if let Some(result_slice) = result.as_slice_mut() {
                    const CHUNK_SIZE: usize = 4096; // Smaller chunks for exp computation
                    result_slice
                        .par_chunks_mut(CHUNK_SIZE)
                        .zip(data_slice.par_chunks(CHUNK_SIZE))
                        .for_each(|(output_chunk, input_chunk)| {
                            for (out, &inp) in output_chunk.iter_mut().zip(input_chunk.iter()) {
                                *out = one / (one + (-inp).exp());
                            }
                        });
                    result
                } else {
                    arr.mapv(|v| one / (one + (-v).exp()))
                }
            } else {
                arr.mapv(|v| one / (one + (-v).exp()))
            }
        }
        ActivationStrategy::Approximation => {
            registry.record_approximation();
            // Use fast approximation for very large arrays
            if std::any::type_name::<T>() == "f32" {
                arr.mapv(|v| {
                    let x = v.to_f32().unwrap_or(0.0);
                    let approx = simd::fast_sigmoid_approx(x);
                    T::from(approx).unwrap_or(v)
                })
            } else {
                arr.mapv(|v| T::one() / (T::one() + (-v).exp()))
            }
        }
        _ => {
            // Sequential implementation
            arr.mapv(|v| T::one() / (T::one() + (-v).exp()))
        }
    };

    // Record performance metrics
    let duration = start_time.elapsed();
    registry.record_function("sigmoid", total_elements, duration.as_nanos() as u64);

    Ok(result)
}

/// Optimized sigmoid implementation for floating-point types
pub fn sigmoid_vectorized<T>(arr: &ArrayD<T>) -> ArrayD<T>
where
    T: Float + Send + Sync,
{
    let one = T::one();
    let total_elements = arr.len();

    if total_elements > super::strategy::PARALLEL_THRESHOLD {
        // Use parallel processing for large arrays
        if let Some(data_slice) = arr.as_slice() {
            let mut result = ArrayD::zeros(arr.raw_dim());
            if let Some(result_slice) = result.as_slice_mut() {
                result_slice
                    .par_iter_mut()
                    .zip(data_slice.par_iter())
                    .for_each(|(out, &input)| {
                        *out = one / (one + (-input).exp());
                    });
                return result;
            }
        }
    }

    // Fallback to standard mapv
    arr.mapv(|v| one / (one + (-v).exp()))
}

/// Optimized tanh implementation for floating-point types
pub fn tanh_vectorized<T>(arr: &ArrayD<T>) -> ArrayD<T>
where
    T: Float + Send + Sync,
{
    let total_elements = arr.len();

    if total_elements > super::strategy::PARALLEL_THRESHOLD {
        // Use parallel processing for large arrays
        if let Some(data_slice) = arr.as_slice() {
            let mut result = ArrayD::zeros(arr.raw_dim());
            if let Some(result_slice) = result.as_slice_mut() {
                result_slice
                    .par_iter_mut()
                    .zip(data_slice.par_iter())
                    .for_each(|(out, &input)| {
                        *out = input.tanh();
                    });
                return result;
            }
        }
    }

    // Fallback to standard mapv
    arr.mapv(|v| v.tanh())
}

/// Sequential GELU implementation for f32
pub fn gelu_sequential_f32(arr: &ArrayD<f32>) -> ArrayD<f32> {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    const GELU_CONST: f32 = 0.044715;

    arr.mapv(|x| {
        let x3 = x * x * x;
        let inner = SQRT_2_OVER_PI * (x + GELU_CONST * x3);
        0.5 * x * (1.0 + inner.tanh())
    })
}

/// Parallel GELU implementation for f32
pub fn gelu_parallel_f32(arr: &ArrayD<f32>) -> ArrayD<f32> {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    const GELU_CONST: f32 = 0.044715;

    if let Some(data_slice) = arr.as_slice() {
        let mut result = ArrayD::zeros(arr.raw_dim());
        if let Some(result_slice) = result.as_slice_mut() {
            result_slice
                .par_iter_mut()
                .zip(data_slice.par_iter())
                .for_each(|(out, &x)| {
                    let x3 = x * x * x;
                    let inner = SQRT_2_OVER_PI * (x + GELU_CONST * x3);
                    *out = 0.5 * x * (1.0 + inner.tanh());
                });
            result
        } else {
            gelu_sequential_f32(arr)
        }
    } else {
        gelu_sequential_f32(arr)
    }
}
