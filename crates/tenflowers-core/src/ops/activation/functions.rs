use crate::{Result, Tensor, TensorError};
use scirs2_core::numeric::{Float, FromPrimitive, Zero};
use std::time::Instant;

use super::core::get_activation_registry;
use super::implementations::{
    gelu_parallel_f32, gelu_sequential_f32, sigmoid_vectorized, tanh_vectorized,
    ultra_relu_vectorized, ultra_sigmoid_vectorized,
};
use super::simd;
use super::strategy::{select_activation_strategy, ActivationStrategy};

/// Ultra-performance ReLU implementation for f32 with comprehensive optimizations
pub fn relu_f32(x: &Tensor<f32>) -> Result<Tensor<f32>> {
    let registry = get_activation_registry();
    let start_time = Instant::now();

    let result = match &x.storage {
        crate::tensor::TensorStorage::Cpu(arr) => {
            let optimized_result = ultra_relu_vectorized(arr)?;
            Ok(Tensor::from_array(optimized_result))
        }
        #[cfg(feature = "gpu")]
        crate::tensor::TensorStorage::Gpu(gpu_buffer) => {
            registry.record_gpu();
            let result_gpu = crate::gpu::ops::execute_activation_op(
                gpu_buffer,
                crate::gpu::ops::ActivationOp::ReLU,
            )?;

            Ok(Tensor::from_gpu_buffer(result_gpu, x.shape().clone()))
        }
    };

    // Record overall function performance
    let duration = start_time.elapsed();
    registry.record_function("relu_f32", x.shape().size(), duration.as_nanos() as u64);

    result
}

/// Ultra-performance ReLU implementation for f64
pub fn relu_f64(x: &Tensor<f64>) -> Result<Tensor<f64>> {
    let registry = get_activation_registry();
    let start_time = Instant::now();

    let result = match &x.storage {
        crate::tensor::TensorStorage::Cpu(arr) => {
            let optimized_result = ultra_relu_vectorized(arr)?;
            Ok(Tensor::from_array(optimized_result))
        }
        #[cfg(feature = "gpu")]
        crate::tensor::TensorStorage::Gpu(gpu_buffer) => {
            registry.record_gpu();
            let result_gpu = crate::gpu::ops::execute_activation_op(
                gpu_buffer,
                crate::gpu::ops::ActivationOp::ReLU,
            )?;

            Ok(Tensor::from_gpu_buffer(result_gpu, x.shape().clone()))
        }
    };

    // Record performance metrics
    let duration = start_time.elapsed();
    registry.record_function("relu_f64", x.shape().size(), duration.as_nanos() as u64);

    result
}

/// Ultra-performance sigmoid implementation for f32
pub fn sigmoid_f32(x: &Tensor<f32>) -> Result<Tensor<f32>> {
    let registry = get_activation_registry();
    let start_time = Instant::now();

    let result = match &x.storage {
        crate::tensor::TensorStorage::Cpu(arr) => {
            let optimized_result = ultra_sigmoid_vectorized(arr)?;
            Ok(Tensor::from_array(optimized_result))
        }
        #[cfg(feature = "gpu")]
        crate::tensor::TensorStorage::Gpu(gpu_buffer) => {
            registry.record_gpu();
            let result_gpu = crate::gpu::ops::execute_activation_op(
                gpu_buffer,
                crate::gpu::ops::ActivationOp::Sigmoid,
            )?;

            let mut result = Tensor::from_gpu_buffer(result_gpu, x.shape().clone());
            result.set_requires_grad(x.requires_grad());
            Ok(result)
        }
    };

    let duration = start_time.elapsed();
    registry.record_function("sigmoid_f32", x.shape().size(), duration.as_nanos() as u64);

    result
}

/// High-performance sigmoid implementation for f64
pub fn sigmoid_f64(x: &Tensor<f64>) -> Result<Tensor<f64>> {
    match &x.storage {
        crate::tensor::TensorStorage::Cpu(arr) => {
            let result = sigmoid_vectorized(arr);
            Ok(Tensor::from_array(result))
        }
        #[cfg(feature = "gpu")]
        crate::tensor::TensorStorage::Gpu(gpu_buffer) => {
            let result_gpu = crate::gpu::ops::execute_activation_op(
                gpu_buffer,
                crate::gpu::ops::ActivationOp::Sigmoid,
            )?;

            let mut result = Tensor::from_gpu_buffer(result_gpu, x.shape().clone());
            result.set_requires_grad(x.requires_grad());
            Ok(result)
        }
    }
}

/// High-performance tanh implementation for f32
pub fn tanh_f32(x: &Tensor<f32>) -> Result<Tensor<f32>> {
    match &x.storage {
        crate::tensor::TensorStorage::Cpu(arr) => {
            let result = tanh_vectorized(arr);
            Ok(Tensor::from_array(result))
        }
        #[cfg(feature = "gpu")]
        crate::tensor::TensorStorage::Gpu(gpu_buffer) => {
            let result_gpu = crate::gpu::ops::execute_activation_op(
                gpu_buffer,
                crate::gpu::ops::ActivationOp::Tanh,
            )?;

            let mut result = Tensor::from_gpu_buffer(result_gpu, x.shape().clone());
            result.set_requires_grad(x.requires_grad());
            Ok(result)
        }
    }
}

/// High-performance tanh implementation for f64
pub fn tanh_f64(x: &Tensor<f64>) -> Result<Tensor<f64>> {
    match &x.storage {
        crate::tensor::TensorStorage::Cpu(arr) => {
            let result = tanh_vectorized(arr);
            Ok(Tensor::from_array(result))
        }
        #[cfg(feature = "gpu")]
        crate::tensor::TensorStorage::Gpu(gpu_buffer) => {
            let result_gpu = crate::gpu::ops::execute_activation_op(
                gpu_buffer,
                crate::gpu::ops::ActivationOp::Tanh,
            )?;

            let mut result = Tensor::from_gpu_buffer(result_gpu, x.shape().clone());
            result.set_requires_grad(x.requires_grad());
            Ok(result)
        }
    }
}

pub fn relu<T>(x: &Tensor<T>) -> Result<Tensor<T>>
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
    match &x.storage {
        crate::tensor::TensorStorage::Cpu(arr) => {
            let zero = T::zero();
            let result = arr.mapv(|v| if v > zero { v } else { zero });
            Ok(Tensor::from_array(result))
        }
        #[cfg(feature = "gpu")]
        crate::tensor::TensorStorage::Gpu(gpu_buffer) => {
            let result_gpu = crate::gpu::ops::execute_activation_op(
                gpu_buffer,
                crate::gpu::ops::ActivationOp::ReLU,
            )?;

            Ok(Tensor::from_gpu_buffer(result_gpu, x.shape().clone()))
        }
    }
}

pub fn sigmoid<T>(x: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Float + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    match &x.storage {
        crate::tensor::TensorStorage::Cpu(arr) => {
            let one = T::one();
            let result = arr.mapv(|v| one / (one + (-v).exp()));
            Ok(Tensor::from_array(result))
        }
        #[cfg(feature = "gpu")]
        crate::tensor::TensorStorage::Gpu(gpu_buffer) => {
            let result_gpu = crate::gpu::ops::execute_activation_op(
                gpu_buffer,
                crate::gpu::ops::ActivationOp::Sigmoid,
            )?;

            let mut result = Tensor::from_gpu_buffer(result_gpu, x.shape().clone());
            result.set_requires_grad(x.requires_grad());
            Ok(result)
        }
    }
}

pub fn tanh<T>(x: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Float + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    match &x.storage {
        crate::tensor::TensorStorage::Cpu(arr) => {
            let result = arr.mapv(|v| v.tanh());
            Ok(Tensor::from_array(result))
        }
        #[cfg(feature = "gpu")]
        crate::tensor::TensorStorage::Gpu(gpu_buffer) => {
            let result_gpu = crate::gpu::ops::execute_activation_op(
                gpu_buffer,
                crate::gpu::ops::ActivationOp::Tanh,
            )?;

            let mut result = Tensor::from_gpu_buffer(result_gpu, x.shape().clone());
            result.set_requires_grad(x.requires_grad());
            Ok(result)
        }
    }
}

pub fn mish<T>(x: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Float + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    match &x.storage {
        crate::tensor::TensorStorage::Cpu(arr) => {
            let result = arr.mapv(|v| {
                let softplus = (T::one() + v.exp()).ln();
                v * softplus.tanh()
            });
            Ok(Tensor::from_array(result))
        }
        #[cfg(feature = "gpu")]
        crate::tensor::TensorStorage::Gpu(_gpu_buffer) => {
            // No native GPU kernel exists for Mish (gpu/ops/activation_ops.rs's
            // UnaryOp only covers ReLU/Sigmoid/Tanh), so the old f32 "fast path"
            // above always errored inside execute_activation_op and was dead
            // code. Use the same host round-trip fallback uniformly for all T.
            let cpu_tensor = x.to_cpu()?;
            let result = mish(&cpu_tensor)?;
            result.to_device(x.device().clone())
        }
    }
}

pub fn softmax<T>(x: &Tensor<T>, axis: Option<i32>) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Float
        + std::ops::Sub<Output = T>
        + std::ops::Add<Output = T>
        + std::ops::Div<Output = T>
        + std::iter::Sum
        + Send
        + Sync
        + bytemuck::Pod,
{
    match &x.storage {
        crate::tensor::TensorStorage::Cpu(arr) => {
            use scirs2_core::ndarray::Axis;

            // Default to last axis if not specified
            let ndim = arr.ndim();
            let axis = axis.unwrap_or(-1);
            let axis = if axis < 0 {
                (ndim as i32 + axis) as usize
            } else {
                axis as usize
            };

            if axis >= ndim {
                return Err(TensorError::InvalidAxis {
                    operation: "softmax".to_string(),
                    axis: axis as i32,
                    ndim,
                    context: None,
                });
            }

            // Simpler implementation that avoids complex indexing
            // First subtract max for numerical stability
            let max_arr = arr.map_axis(Axis(axis), |view| {
                view.iter().cloned().fold(T::neg_infinity(), T::max)
            });

            // Expand max_arr to match original dimensions
            let mut shape_vec = arr.shape().to_vec();
            shape_vec[axis] = 1;
            let max_expanded = max_arr
                .into_shape_with_order(shape_vec.as_slice())
                .map_err(|e| TensorError::invalid_shape_simple(e.to_string()))?;

            // Compute exp(x - max)
            let exp_arr = arr.clone()
                - &max_expanded
                    .broadcast(arr.dim())
                    .expect("max array must be broadcastable to input shape");
            let exp_arr = exp_arr.mapv(|x| x.exp());

            // Sum along axis
            let sum_arr = exp_arr.sum_axis(Axis(axis));

            // Expand sum to match dimensions
            let sum_expanded = sum_arr
                .into_shape_with_order(shape_vec.as_slice())
                .map_err(|e| TensorError::invalid_shape_simple(e.to_string()))?;

            // Divide to get softmax
            let result = exp_arr
                / &sum_expanded
                    .broadcast(arr.dim())
                    .expect("sum array must be broadcastable to input shape");

            Ok(Tensor::from_array(result))
        }
        #[cfg(feature = "gpu")]
        crate::tensor::TensorStorage::Gpu(gpu_buffer) => {
            // Implement GPU softmax using axis-specific reductions
            if std::any::type_name::<T>() == "f32" {
                super::gpu::softmax_gpu_f32(x, axis)
            } else {
                // Fallback to CPU for non-f32 types
                let cpu_tensor = x.to_cpu()?;
                let result = softmax(&cpu_tensor, axis)?;
                result.to_device(x.device().clone())
            }
        }
    }
}

/// GELU (Gaussian Error Linear Unit) activation function
/// GELU(x) = x * Φ(x) where Φ is the standard Gaussian CDF
/// Approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
pub fn gelu<T>(x: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Float + Send + Sync + 'static + bytemuck::Pod,
{
    match &x.storage {
        crate::tensor::TensorStorage::Cpu(arr) => {
            let sqrt_2_over_pi = T::from(0.797_884_608)
                .expect("constant 0.797884608 must be convertible to float type"); // √(2/π)
            let coeff =
                T::from(0.044715).expect("constant 0.044715 must be convertible to float type");
            let half = T::from(0.5).expect("constant 0.5 must be convertible to float type");
            let one = T::one();

            let result = arr.mapv(|x| {
                let x_cubed = x * x * x;
                let inner = sqrt_2_over_pi * (x + coeff * x_cubed);
                half * x * (one + inner.tanh())
            });

            Ok(Tensor::from_array(result))
        }
        #[cfg(feature = "gpu")]
        crate::tensor::TensorStorage::Gpu(_gpu_buffer) => {
            // No native GPU kernel exists for GELU; round-trip through the
            // host and delegate to the known-correct CPU implementation
            // (mirrors mish/softmax's existing non-f32 fallback pattern).
            let cpu_tensor = x.to_cpu()?;
            let result = gelu(&cpu_tensor)?;
            result.to_device(x.device().clone())
        }
    }
}

/// Ultra-performance GELU implementation (Gaussian Error Linear Unit)
pub fn gelu_f32(x: &Tensor<f32>) -> Result<Tensor<f32>> {
    let registry = get_activation_registry();
    let start_time = Instant::now();

    let result = match &x.storage {
        crate::tensor::TensorStorage::Cpu(arr) => {
            let total_elements = arr.len();
            let strategy = select_activation_strategy(total_elements, true);

            let computed_result = match strategy {
                ActivationStrategy::Simd => {
                    if arr.is_standard_layout() {
                        if let Some(input_slice) = arr.as_slice() {
                            let mut output_data = vec![0.0f32; input_slice.len()];
                            match simd::simd_gelu_f32(input_slice, &mut output_data) {
                                Ok(()) => {
                                    registry.record_simd();
                                    scirs2_core::ndarray::ArrayD::from_shape_vec(
                                        arr.raw_dim(),
                                        output_data,
                                    )
                                    .map_err(|e| {
                                        TensorError::invalid_argument(format!(
                                            "SIMD GELU shape error: {}",
                                            e
                                        ))
                                    })?
                                }
                                Err(_) => {
                                    // Fallback to sequential
                                    gelu_sequential_f32(arr)
                                }
                            }
                        } else {
                            gelu_sequential_f32(arr)
                        }
                    } else {
                        gelu_sequential_f32(arr)
                    }
                }
                ActivationStrategy::Parallel => {
                    registry.record_parallel();
                    gelu_parallel_f32(arr)
                }
                _ => gelu_sequential_f32(arr),
            };

            Ok(Tensor::from_array(computed_result))
        }
        #[cfg(feature = "gpu")]
        crate::tensor::TensorStorage::Gpu(_gpu_buffer) => {
            registry.record_gpu();
            // No native GPU kernel exists for GELU; round-trip through the
            // host and delegate to the known-correct CPU implementation
            // (mirrors mish/softmax's existing non-f32 fallback pattern).
            let cpu_tensor = x.to_cpu()?;
            let result = gelu_f32(&cpu_tensor)?;
            result.to_device(x.device().clone())
        }
    };

    let duration = start_time.elapsed();
    registry.record_function("gelu_f32", x.shape().size(), duration.as_nanos() as u64);

    result
}

/// Swish activation function (also known as SiLU - Sigmoid Linear Unit)
/// Swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
pub fn swish<T>(x: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Float + Send + Sync + 'static + bytemuck::Pod,
{
    match &x.storage {
        crate::tensor::TensorStorage::Cpu(arr) => {
            let one = T::one();
            let result = arr.mapv(|x| {
                let sigmoid_x = one / (one + (-x).exp());
                x * sigmoid_x
            });

            Ok(Tensor::from_array(result))
        }
        #[cfg(feature = "gpu")]
        crate::tensor::TensorStorage::Gpu(_gpu_buffer) => {
            // No native GPU kernel exists for Swish; round-trip through the
            // host and delegate to the known-correct CPU implementation
            // (mirrors mish/softmax's existing non-f32 fallback pattern).
            let cpu_tensor = x.to_cpu()?;
            let result = swish(&cpu_tensor)?;
            result.to_device(x.device().clone())
        }
    }
}

/// ELU (Exponential Linear Unit) activation function
/// ELU(x) = x if x > 0, α * (exp(x) - 1) if x <= 0
pub fn elu<T>(x: &Tensor<T>, alpha: T) -> Result<Tensor<T>>
where
    T: Clone + Default + Float + PartialOrd + Send + Sync + 'static + bytemuck::Pod,
{
    match &x.storage {
        crate::tensor::TensorStorage::Cpu(arr) => {
            let zero = T::zero();
            let one = T::one();

            let result = arr.mapv(|x| if x > zero { x } else { alpha * (x.exp() - one) });

            Ok(Tensor::from_array(result))
        }
        #[cfg(feature = "gpu")]
        crate::tensor::TensorStorage::Gpu(_gpu_buffer) => {
            // No native GPU kernel exists for ELU; round-trip through the
            // host and delegate to the known-correct CPU implementation
            // (mirrors mish/softmax's existing non-f32 fallback pattern).
            let cpu_tensor = x.to_cpu()?;
            let result = elu(&cpu_tensor, alpha)?;
            result.to_device(x.device().clone())
        }
    }
}

/// LeakyReLU activation function
/// LeakyReLU(x) = max(αx, x) where α is typically 0.01
pub fn leaky_relu<T>(x: &Tensor<T>, alpha: T) -> Result<Tensor<T>>
where
    T: Clone + Default + Float + PartialOrd + Send + Sync + 'static + bytemuck::Pod,
{
    match &x.storage {
        crate::tensor::TensorStorage::Cpu(arr) => {
            let zero = T::zero();

            let result = arr.mapv(|x| if x > zero { x } else { alpha * x });

            Ok(Tensor::from_array(result))
        }
        #[cfg(feature = "gpu")]
        crate::tensor::TensorStorage::Gpu(_gpu_buffer) => {
            // No native GPU kernel exists for LeakyReLU; round-trip through
            // the host and delegate to the known-correct CPU implementation
            // (mirrors mish/softmax's existing non-f32 fallback pattern).
            let cpu_tensor = x.to_cpu()?;
            let result = leaky_relu(&cpu_tensor, alpha)?;
            result.to_device(x.device().clone())
        }
    }
}

/// HardSwish activation function
/// HardSwish(x) = x * ReLU6(x + 3) / 6
/// where ReLU6(x) = min(max(x, 0), 6)
pub fn hard_swish<T>(x: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Float
        + PartialOrd
        + Send
        + Sync
        + bytemuck::Pod
        + bytemuck::Zeroable
        + 'static,
{
    match &x.storage {
        crate::tensor::TensorStorage::Cpu(arr) => {
            let zero = T::zero();
            let three = T::from(3).expect("constant 3 must be convertible to float type");
            let six = T::from(6).expect("constant 6 must be convertible to float type");

            let result = arr.mapv(|x| {
                let x_plus_3 = x + three;
                let relu6_val = if x_plus_3 < zero {
                    zero
                } else if x_plus_3 > six {
                    six
                } else {
                    x_plus_3
                };
                x * relu6_val / six
            });

            Ok(Tensor::from_array(result))
        }
        #[cfg(feature = "gpu")]
        crate::tensor::TensorStorage::Gpu(_gpu_buffer) => {
            // No native GPU kernel exists for HardSwish; round-trip through
            // the host and delegate to the known-correct CPU implementation
            // (mirrors mish/softmax's existing non-f32 fallback pattern).
            let cpu_tensor = x.to_cpu()?;
            let result = hard_swish(&cpu_tensor)?;
            result.to_device(x.device().clone())
        }
    }
}

/// PReLU (Parametric ReLU) activation function
/// PReLU(x) = max(0, x) + α * min(0, x)
/// where α is a learnable parameter (different for each channel)
pub fn prelu<T>(x: &Tensor<T>, alpha: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Float
        + PartialOrd
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    let x_shape = x.shape().dims();
    let alpha_shape = alpha.shape().dims();

    // Alpha should be either scalar or match the channel dimension
    if alpha_shape.len() != 1 && alpha_shape != [1] {
        return Err(TensorError::invalid_shape_simple(format!(
            "PReLU alpha must be 1D, got shape {alpha_shape:?}"
        )));
    }

    match (&x.storage, &alpha.storage) {
        (
            crate::tensor::TensorStorage::Cpu(x_arr),
            crate::tensor::TensorStorage::Cpu(alpha_arr),
        ) => {
            let zero = T::zero();

            // Handle broadcasting of alpha
            let result = if alpha_shape == [1] {
                // Scalar alpha
                let alpha_val = alpha_arr[[0]];
                x_arr.mapv(|x| if x > zero { x } else { alpha_val * x })
            } else {
                // Channel-wise alpha (assuming NCHW format)
                if x_shape.len() < 2 {
                    return Err(TensorError::invalid_shape_simple(
                        "PReLU with channel-wise alpha requires at least 2D input ".to_string(),
                    ));
                }

                let channels = x_shape[1];
                if alpha_shape[0] != channels {
                    return Err(TensorError::shape_mismatch(
                        "prelu",
                        &format!("alpha channels = {channels}"),
                        &format!("alpha channels = {}", alpha_shape[0]),
                    ));
                }

                let mut result = x_arr.clone();
                for (mut slice, &alpha_val) in result
                    .axis_iter_mut(scirs2_core::ndarray::Axis(1))
                    .zip(alpha_arr.iter())
                {
                    slice.mapv_inplace(|x| if x <= zero { alpha_val * x } else { x });
                }
                result
            };

            Ok(Tensor::from_array(result))
        }
        #[cfg(feature = "gpu")]
        (
            crate::tensor::TensorStorage::Gpu(_x_buffer),
            crate::tensor::TensorStorage::Gpu(_alpha_buffer),
        ) => {
            // Use GPU binary operation for PReLU
            use crate::ops::binary::{binary_op, PReLUOp};
            binary_op(x, alpha, PReLUOp)
        }
        #[cfg(feature = "gpu")]
        _ => Err(TensorError::device_mismatch(
            "prelu",
            &x.device().to_string(),
            &alpha.device().to_string(),
        )),
    }
}

/// ReLU6 activation: min(max(x, 0), 6)
pub fn relu6<T>(x: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + PartialOrd
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable
        + FromPrimitive,
{
    match &x.storage {
        crate::tensor::TensorStorage::Cpu(arr) => {
            let zero = T::zero();
            let six = T::from_u32(6).unwrap_or_default();
            let result = arr.mapv(|v| {
                if v <= zero {
                    zero
                } else if v >= six {
                    six
                } else {
                    v
                }
            });
            Ok(Tensor::from_array(result))
        }
        #[cfg(feature = "gpu")]
        crate::tensor::TensorStorage::Gpu(_gpu_buffer) => {
            // For now, fall back to CPU implementation
            let cpu_tensor = x.to_cpu()?;
            relu6(&cpu_tensor)
        }
    }
}

/// Hard Swish activation: x * relu6(x + 3) / 6
pub fn hardswish<T>(x: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + PartialOrd
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable
        + FromPrimitive
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>,
{
    match &x.storage {
        crate::tensor::TensorStorage::Cpu(arr) => {
            let zero = T::zero();
            let three = T::from_u32(3).unwrap_or_default();
            let six = T::from_u32(6).unwrap_or_default();
            let result = arr.mapv(|v| {
                let relu6_val = if v + three <= zero {
                    zero
                } else if v + three >= six {
                    six
                } else {
                    v + three
                };
                v * relu6_val / six
            });
            Ok(Tensor::from_array(result))
        }
        #[cfg(feature = "gpu")]
        crate::tensor::TensorStorage::Gpu(_gpu_buffer) => {
            // For now, fall back to CPU implementation
            let cpu_tensor = x.to_cpu()?;
            hardswish(&cpu_tensor)
        }
    }
}

/// Log Softmax activation: log(softmax(x))
pub fn log_softmax<T>(x: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + PartialOrd
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable
        + scirs2_core::num_traits::Float,
{
    // log_softmax(x) = x - log(sum(exp(x)))
    // This is numerically stable implementation
    match &x.storage {
        crate::tensor::TensorStorage::Cpu(arr) => {
            // Find max for numerical stability
            let max_val = arr.fold(
                T::neg_infinity(),
                |acc, &val| {
                    if val > acc {
                        val
                    } else {
                        acc
                    }
                },
            );

            // Compute log_softmax
            let sum_exp = arr.fold(T::zero(), |acc, &val| acc + (val - max_val).exp());
            let log_sum_exp = sum_exp.ln() + max_val;

            let result = arr.mapv(|v| v - log_sum_exp);
            Ok(Tensor::from_array(result))
        }
        #[cfg(feature = "gpu")]
        crate::tensor::TensorStorage::Gpu(_gpu_buffer) => {
            // For now, fall back to CPU implementation
            let cpu_tensor = x.to_cpu()?;
            log_softmax(&cpu_tensor)
        }
    }
}

// GPU delegate correctness tests: verify each Tensor-level activation caller
// that has no native GPU kernel (gelu, gelu_f32, swish, elu, leaky_relu,
// hard_swish, mish) round-trips GPU-resident input through the host and
// delegates to the known-correct CPU implementation, instead of erroring via
// `execute_activation_op` (which only implements ReLU/Sigmoid/Tanh natively).
// A GPU adapter is not guaranteed to be present in every environment that
// builds with `--features gpu`; each test attempts the transfer and skips
// (returns early) if no adapter is available, mirroring the convention in
// `ops/einsum/gpu.rs`.
#[cfg(all(test, feature = "gpu"))]
mod gpu_delegate_tests {
    use super::*;
    use crate::Device;

    #[test]
    fn gpu_gelu_matches_cpu_reference() {
        let data = vec![-2.0f32, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
        let cpu_input = Tensor::<f32>::from_vec(data, &[7]).expect("test: from_vec should succeed");
        let cpu_result = gelu(&cpu_input).expect("test: CPU gelu should succeed");

        let gpu_input = match cpu_input.to(Device::Gpu(0)) {
            Ok(t) => t,
            Err(_) => return, // No GPU adapter available in this environment; skip.
        };
        let gpu_result =
            gelu(&gpu_input).expect("test: GPU gelu should succeed via CPU-delegate fallback");
        assert_eq!(gpu_result.shape().dims(), cpu_result.shape().dims());
        assert_eq!(
            gpu_result.to_vec().expect("test: to_vec should succeed"),
            cpu_result.to_vec().expect("test: to_vec should succeed"),
        );
    }

    #[test]
    fn gpu_gelu_f32_matches_cpu_reference() {
        let data = vec![-2.0f32, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
        let cpu_input = Tensor::<f32>::from_vec(data, &[7]).expect("test: from_vec should succeed");
        let cpu_result = gelu_f32(&cpu_input).expect("test: CPU gelu_f32 should succeed");

        let gpu_input = match cpu_input.to(Device::Gpu(0)) {
            Ok(t) => t,
            Err(_) => return,
        };
        let gpu_result = gelu_f32(&gpu_input)
            .expect("test: GPU gelu_f32 should succeed via CPU-delegate fallback");
        assert_eq!(gpu_result.shape().dims(), cpu_result.shape().dims());
        assert_eq!(
            gpu_result.to_vec().expect("test: to_vec should succeed"),
            cpu_result.to_vec().expect("test: to_vec should succeed"),
        );
    }

    #[test]
    fn gpu_swish_matches_cpu_reference() {
        let data = vec![-2.0f32, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
        let cpu_input = Tensor::<f32>::from_vec(data, &[7]).expect("test: from_vec should succeed");
        let cpu_result = swish(&cpu_input).expect("test: CPU swish should succeed");

        let gpu_input = match cpu_input.to(Device::Gpu(0)) {
            Ok(t) => t,
            Err(_) => return,
        };
        let gpu_result =
            swish(&gpu_input).expect("test: GPU swish should succeed via CPU-delegate fallback");
        assert_eq!(gpu_result.shape().dims(), cpu_result.shape().dims());
        assert_eq!(
            gpu_result.to_vec().expect("test: to_vec should succeed"),
            cpu_result.to_vec().expect("test: to_vec should succeed"),
        );
    }

    #[test]
    fn gpu_elu_matches_cpu_reference() {
        let data = vec![-2.0f32, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
        let cpu_input = Tensor::<f32>::from_vec(data, &[7]).expect("test: from_vec should succeed");
        let cpu_result = elu(&cpu_input, 1.0f32).expect("test: CPU elu should succeed");

        let gpu_input = match cpu_input.to(Device::Gpu(0)) {
            Ok(t) => t,
            Err(_) => return,
        };
        let gpu_result = elu(&gpu_input, 1.0f32)
            .expect("test: GPU elu should succeed via CPU-delegate fallback");
        assert_eq!(gpu_result.shape().dims(), cpu_result.shape().dims());
        assert_eq!(
            gpu_result.to_vec().expect("test: to_vec should succeed"),
            cpu_result.to_vec().expect("test: to_vec should succeed"),
        );
    }

    #[test]
    fn gpu_leaky_relu_matches_cpu_reference() {
        let data = vec![-2.0f32, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
        let cpu_input = Tensor::<f32>::from_vec(data, &[7]).expect("test: from_vec should succeed");
        let cpu_result =
            leaky_relu(&cpu_input, 0.01f32).expect("test: CPU leaky_relu should succeed");

        let gpu_input = match cpu_input.to(Device::Gpu(0)) {
            Ok(t) => t,
            Err(_) => return,
        };
        let gpu_result = leaky_relu(&gpu_input, 0.01f32)
            .expect("test: GPU leaky_relu should succeed via CPU-delegate fallback");
        assert_eq!(gpu_result.shape().dims(), cpu_result.shape().dims());
        assert_eq!(
            gpu_result.to_vec().expect("test: to_vec should succeed"),
            cpu_result.to_vec().expect("test: to_vec should succeed"),
        );
    }

    #[test]
    fn gpu_hard_swish_matches_cpu_reference() {
        let data = vec![-4.0f32, -2.0, -0.5, 0.0, 0.5, 2.0, 4.0];
        let cpu_input = Tensor::<f32>::from_vec(data, &[7]).expect("test: from_vec should succeed");
        let cpu_result = hard_swish(&cpu_input).expect("test: CPU hard_swish should succeed");

        let gpu_input = match cpu_input.to(Device::Gpu(0)) {
            Ok(t) => t,
            Err(_) => return,
        };
        let gpu_result = hard_swish(&gpu_input)
            .expect("test: GPU hard_swish should succeed via CPU-delegate fallback");
        assert_eq!(gpu_result.shape().dims(), cpu_result.shape().dims());
        assert_eq!(
            gpu_result.to_vec().expect("test: to_vec should succeed"),
            cpu_result.to_vec().expect("test: to_vec should succeed"),
        );
    }

    #[test]
    fn gpu_mish_matches_cpu_reference_no_dead_fast_path() {
        // Also guards against regressing the deleted dead f32 "fast path":
        // mish must now use the uniform readback+delegate pattern for f32 too,
        // since execute_activation_op has no real Mish kernel.
        let data = vec![-2.0f32, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
        let cpu_input = Tensor::<f32>::from_vec(data, &[7]).expect("test: from_vec should succeed");
        let cpu_result = mish(&cpu_input).expect("test: CPU mish should succeed");

        let gpu_input = match cpu_input.to(Device::Gpu(0)) {
            Ok(t) => t,
            Err(_) => return,
        };
        let gpu_result =
            mish(&gpu_input).expect("test: GPU mish should succeed via CPU-delegate fallback");
        assert_eq!(gpu_result.shape().dims(), cpu_result.shape().dims());
        assert_eq!(
            gpu_result.to_vec().expect("test: to_vec should succeed"),
            cpu_result.to_vec().expect("test: to_vec should succeed"),
        );
    }
}
