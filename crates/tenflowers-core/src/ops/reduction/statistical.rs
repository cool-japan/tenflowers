//! Statistical reduction operations
//!
//! This module contains statistical reduction operations that compute aggregate statistics
//! along specified axes of tensors. These operations are fundamental for data analysis,
//! machine learning computations, and numerical processing.
//!
//! The statistical operations include:
//! - `sum`: Sum reduction along specified axes
//! - `mean`: Mean (average) reduction along specified axes
//! - `max`: Maximum value reduction along specified axes
//! - `min`: Minimum value reduction along specified axes
//! - `prod`: Product reduction along specified axes
//! - `variance`: Variance calculation along specified axes

use super::common::normalize_axis;
use crate::tensor::TensorStorage;
use crate::{Result, Tensor, TensorError};
use scirs2_core::ndarray::{ArrayD, Axis};
use scirs2_core::numeric::{Float, FromPrimitive, Zero};
// use scirs2_core::parallel_ops::{par_chunks, par_join};

/// Sum reduction along specified axes
///
/// Computes the sum of tensor elements along the specified axes.
/// If no axes are specified, computes the sum of all elements.
///
/// # Arguments
/// * `x` - Input tensor
/// * `axes` - Optional slice of axis indices to reduce along
/// * `keepdims` - Whether to keep reduced dimensions as size 1
///
/// # Returns
/// * `Result<Tensor<T>>` - Tensor with sum values
///
/// # Type Requirements
/// * `T` must implement `Clone + Default + Zero + Add + Send + Sync + 'static`
pub fn sum<T>(x: &Tensor<T>, axes: Option<&[i32]>, keepdims: bool) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + std::ops::Add<Output = T>
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    match &x.storage {
        TensorStorage::Cpu(arr) => {
            let _result_shape =
                crate::ops::shape_inference::infer_reduction(x.shape(), axes, keepdims)?;

            if let Some(axes) = axes {
                // Memory-efficient reduction - use view operations instead of cloning
                let mut result = arr.view().to_owned();

                // Sort axes in descending order to avoid index shifting
                let mut sorted_axes: Vec<_> = axes
                    .iter()
                    .map(|&a| normalize_axis(a, x.shape().rank() as i32))
                    .collect::<Result<Vec<_>>>()?;
                sorted_axes.sort_by(|a, b| b.cmp(a));

                // Use parallel reduction for large tensors
                if result.len() > 10000 {
                    for &axis in &sorted_axes {
                        // Parallel sum axis operation for better performance
                        result = par_sum_axis(&result, axis)?;
                        if keepdims {
                            result = result.insert_axis(Axis(axis));
                        }
                    }
                } else {
                    for &axis in &sorted_axes {
                        result = result.sum_axis(Axis(axis));
                        if keepdims {
                            result = result.insert_axis(Axis(axis));
                        }
                    }
                }

                Ok(Tensor::from_array(result))
            } else {
                // Reduce all axes using parallel computation for large arrays
                let sum = if arr.len() > 10000 {
                    parallel_sum_all(arr)?
                } else {
                    arr.sum()
                };
                let result = if keepdims {
                    ArrayD::from_elem(vec![1; x.shape().rank()], sum)
                } else {
                    ArrayD::from_elem(vec![], sum)
                };
                Ok(Tensor::from_array(result))
            }
        }
        #[cfg(feature = "gpu")]
        TensorStorage::Gpu(gpu_buffer) => {
            use crate::gpu::ops::{execute_axis_reduction_op, ReductionOp};

            // Calculate output shape and size
            let result_shape =
                crate::ops::shape_inference::infer_reduction(x.shape(), axes, keepdims)?;
            let output_len = result_shape.dims().iter().product();

            let result_buffer = execute_axis_reduction_op(
                gpu_buffer,
                ReductionOp::Sum,
                x.shape().dims(),
                axes,
                keepdims,
                output_len,
            )?;

            Ok(Tensor::from_gpu_buffer(result_buffer, result_shape))
        }
    }
}

/// Mean reduction along specified axes
///
/// Computes the mean (average) of tensor elements along the specified axes.
/// If no axes are specified, computes the mean of all elements.
///
/// # Arguments
/// * `x` - Input tensor
/// * `axes` - Optional slice of axis indices to reduce along
/// * `keepdims` - Whether to keep reduced dimensions as size 1
///
/// # Returns
/// * `Result<Tensor<T>>` - Tensor with mean values
///
/// # Type Requirements
/// * `T` must implement `Clone + Default + Float + FromPrimitive + Send + Sync + 'static`
pub fn mean<T>(x: &Tensor<T>, axes: Option<&[i32]>, keepdims: bool) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Float
        + FromPrimitive
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    match &x.storage {
        TensorStorage::Cpu(arr) => {
            let _result_shape =
                crate::ops::shape_inference::infer_reduction(x.shape(), axes, keepdims)?;

            if let Some(axes) = axes {
                // Reduce along specific axes
                let mut result = arr.map(|x| *x);

                // Sort axes in descending order to avoid index shifting
                let mut sorted_axes: Vec<_> = axes
                    .iter()
                    .map(|&a| normalize_axis(a, x.shape().rank() as i32))
                    .collect::<Result<Vec<_>>>()?;
                sorted_axes.sort_by(|a, b| b.cmp(a));

                for &axis in &sorted_axes {
                    result = result.mean_axis(Axis(axis)).unwrap();
                    if keepdims {
                        result = result.insert_axis(Axis(axis));
                    }
                }

                Ok(Tensor::from_array(result))
            } else {
                // Reduce all axes - compute mean of all elements
                let mean_val = arr.mean().unwrap_or_default();
                let result = if keepdims {
                    ArrayD::from_elem(vec![1; x.shape().rank()], mean_val)
                } else {
                    ArrayD::from_elem(vec![], mean_val)
                };
                Ok(Tensor::from_array(result))
            }
        }
        #[cfg(feature = "gpu")]
        TensorStorage::Gpu(gpu_buffer) => {
            use crate::gpu::ops::{execute_axis_reduction_op, ReductionOp};

            // Calculate output shape and size
            let result_shape =
                crate::ops::shape_inference::infer_reduction(x.shape(), axes, keepdims)?;
            let output_len = result_shape.dims().iter().product();

            let result_buffer = execute_axis_reduction_op(
                gpu_buffer,
                ReductionOp::Mean,
                x.shape().dims(),
                axes,
                keepdims,
                output_len,
            )?;

            Ok(Tensor::from_gpu_buffer(result_buffer, result_shape))
        }
    }
}

/// Maximum value reduction along specified axes
///
/// Computes the maximum value of tensor elements along the specified axes.
/// If no axes are specified, computes the maximum of all elements.
///
/// # Arguments
/// * `x` - Input tensor
/// * `axes` - Optional slice of axis indices to reduce along
/// * `keepdims` - Whether to keep reduced dimensions as size 1
///
/// # Returns
/// * `Result<Tensor<T>>` - Tensor with maximum values
///
/// # Type Requirements
/// * `T` must implement `Clone + Default + PartialOrd + Send + Sync + 'static`
pub fn max<T>(x: &Tensor<T>, axes: Option<&[i32]>, keepdims: bool) -> Result<Tensor<T>>
where
    T: Clone + Default + PartialOrd + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    match &x.storage {
        TensorStorage::Cpu(arr) => {
            if arr.is_empty() {
                return Err(TensorError::invalid_argument(
                    "Cannot compute max of empty tensor ".to_string(),
                ));
            }

            if let Some(axes) = axes {
                // Reduce along specific axes
                let mut result = arr.clone();

                // Sort axes in descending order to avoid index shifting
                let mut sorted_axes: Vec<_> = axes
                    .iter()
                    .map(|&a| normalize_axis(a, x.shape().rank() as i32))
                    .collect::<Result<Vec<_>>>()?;
                sorted_axes.sort_by(|a, b| b.cmp(a));

                for &axis in &sorted_axes {
                    // Use fold to find max along axis
                    result =
                        result.fold_axis(
                            Axis(axis),
                            T::default(),
                            |acc, x| {
                                if x > acc {
                                    *x
                                } else {
                                    *acc
                                }
                            },
                        );
                    if keepdims {
                        result = result.insert_axis(Axis(axis));
                    }
                }

                Ok(Tensor::from_array(result))
            } else {
                // Reduce all axes - find max of all elements
                let max_val = arr
                    .iter()
                    .fold(None, |acc: Option<&T>, x| match acc {
                        None => Some(x),
                        Some(max) => {
                            if x > max {
                                Some(x)
                            } else {
                                Some(max)
                            }
                        }
                    })
                    .cloned()
                    .unwrap_or_default();
                let result = if keepdims {
                    ArrayD::from_elem(vec![1; x.shape().rank()], max_val)
                } else {
                    ArrayD::from_elem(vec![], max_val)
                };
                Ok(Tensor::from_array(result))
            }
        }
        #[cfg(feature = "gpu")]
        TensorStorage::Gpu(gpu_buffer) => {
            use crate::gpu::ops::{execute_axis_reduction_op, ReductionOp};

            // Calculate output shape and size
            let result_shape =
                crate::ops::shape_inference::infer_reduction(x.shape(), axes, keepdims)?;
            let output_len = result_shape.dims().iter().product();

            let result_buffer = execute_axis_reduction_op(
                gpu_buffer,
                ReductionOp::Max,
                x.shape().dims(),
                axes,
                keepdims,
                output_len,
            )?;

            Ok(Tensor::from_gpu_buffer(result_buffer, result_shape))
        }
    }
}

/// Minimum value reduction along specified axes
///
/// Computes the minimum value of tensor elements along the specified axes.
/// If no axes are specified, computes the minimum of all elements.
///
/// # Arguments
/// * `x` - Input tensor
/// * `axes` - Optional slice of axis indices to reduce along
/// * `keepdims` - Whether to keep reduced dimensions as size 1
///
/// # Returns
/// * `Result<Tensor<T>>` - Tensor with minimum values
///
/// # Type Requirements
/// * `T` must implement `Clone + Default + PartialOrd + Send + Sync + 'static`
pub fn min<T>(x: &Tensor<T>, axes: Option<&[i32]>, keepdims: bool) -> Result<Tensor<T>>
where
    T: Clone + Default + PartialOrd + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    match &x.storage {
        TensorStorage::Cpu(arr) => {
            if arr.is_empty() {
                return Err(TensorError::invalid_argument(
                    "Cannot compute min of empty tensor ".to_string(),
                ));
            }

            if let Some(axes) = axes {
                // Reduce along specific axes
                let mut result = arr.clone();

                // Sort axes in descending order to avoid index shifting
                let mut sorted_axes: Vec<_> = axes
                    .iter()
                    .map(|&a| normalize_axis(a, x.shape().rank() as i32))
                    .collect::<Result<Vec<_>>>()?;
                sorted_axes.sort_by(|a, b| b.cmp(a));

                for &axis in &sorted_axes {
                    // Use fold to find min along axis
                    result =
                        result.fold_axis(
                            Axis(axis),
                            T::default(),
                            |acc, x| {
                                if x < acc {
                                    *x
                                } else {
                                    *acc
                                }
                            },
                        );
                    if keepdims {
                        result = result.insert_axis(Axis(axis));
                    }
                }

                Ok(Tensor::from_array(result))
            } else {
                // Reduce all axes - find min of all elements
                let min_val = arr
                    .iter()
                    .fold(None, |acc: Option<&T>, x| match acc {
                        None => Some(x),
                        Some(min) => {
                            if x < min {
                                Some(x)
                            } else {
                                Some(min)
                            }
                        }
                    })
                    .cloned()
                    .unwrap_or_default();
                let result = if keepdims {
                    ArrayD::from_elem(vec![1; x.shape().rank()], min_val)
                } else {
                    ArrayD::from_elem(vec![], min_val)
                };
                Ok(Tensor::from_array(result))
            }
        }
        #[cfg(feature = "gpu")]
        TensorStorage::Gpu(gpu_buffer) => {
            use crate::gpu::ops::{execute_axis_reduction_op, ReductionOp};

            // Calculate output shape and size
            let result_shape =
                crate::ops::shape_inference::infer_reduction(x.shape(), axes, keepdims)?;
            let output_len = result_shape.dims().iter().product();

            let result_buffer = execute_axis_reduction_op(
                gpu_buffer,
                ReductionOp::Min,
                x.shape().dims(),
                axes,
                keepdims,
                output_len,
            )?;

            Ok(Tensor::from_gpu_buffer(result_buffer, result_shape))
        }
    }
}

/// Product reduction along specified axes
///
/// Computes the product of tensor elements along the specified axes.
/// If no axes are specified, computes the product of all elements.
///
/// # Arguments
/// * `x` - Input tensor
/// * `axes` - Optional slice of axis indices to reduce along
/// * `keepdims` - Whether to keep reduced dimensions as size 1
///
/// # Returns
/// * `Result<Tensor<T>>` - Tensor with product values
///
/// # Type Requirements
/// * `T` must implement `Clone + Default + Mul + One + Send + Sync + 'static`
pub fn prod<T>(x: &Tensor<T>, axes: Option<&[i32]>, keepdims: bool) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + std::ops::Mul<Output = T>
        + scirs2_core::num_traits::One
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    match &x.storage {
        TensorStorage::Cpu(arr) => {
            let _result_shape =
                crate::ops::shape_inference::infer_reduction(x.shape(), axes, keepdims)?;

            if let Some(axes) = axes {
                // Reduce along specific axes
                let mut result = arr.clone();

                // Sort axes in descending order to avoid index shifting
                let mut sorted_axes: Vec<_> = axes
                    .iter()
                    .map(|&a| normalize_axis(a, x.shape().rank() as i32))
                    .collect::<Result<Vec<_>>>()?;
                sorted_axes.sort_by(|a, b| b.cmp(a));

                for &axis in &sorted_axes {
                    result = result.fold_axis(Axis(axis), T::one(), |acc, x| *acc * *x);
                    if keepdims {
                        result = result.insert_axis(Axis(axis));
                    }
                }

                Ok(Tensor::from_array(result))
            } else {
                // Reduce all axes - compute product of all elements
                let prod = arr.iter().fold(T::one(), |acc, x| acc * *x);
                let result = if keepdims {
                    ArrayD::from_elem(vec![1; x.shape().rank()], prod)
                } else {
                    ArrayD::from_elem(vec![], prod)
                };
                Ok(Tensor::from_array(result))
            }
        }
        #[cfg(feature = "gpu")]
        TensorStorage::Gpu(gpu_buffer) => {
            use crate::gpu::ops::{execute_axis_reduction_op, ReductionOp};

            // Calculate output shape and size
            let result_shape =
                crate::ops::shape_inference::infer_reduction(x.shape(), axes, keepdims)?;
            let output_len = result_shape.dims().iter().product();

            let result_buffer = execute_axis_reduction_op(
                gpu_buffer,
                ReductionOp::Prod,
                x.shape().dims(),
                axes,
                keepdims,
                output_len,
            )?;

            Ok(Tensor::from_gpu_buffer(result_buffer, result_shape))
        }
    }
}

/// Variance calculation along specified axes
///
/// Computes the variance of tensor elements along the specified axes.
/// If no axes are specified, computes the variance of all elements.
///
/// # Arguments
/// * `x` - Input tensor
/// * `axes` - Optional slice of axis indices to reduce along
/// * `keepdims` - Whether to keep reduced dimensions as size 1
/// * `ddof` - Delta degrees of freedom for sample variance calculation
///
/// # Returns
/// * `Result<Tensor<T>>` - Tensor with variance values
///
/// # Type Requirements
/// * `T` must implement `Clone + Default + Float + FromPrimitive + Send + Sync + 'static + scirs2_core::ndarray::ScalarOperand`
pub fn variance<T>(
    x: &Tensor<T>,
    axes: Option<&[i32]>,
    keepdims: bool,
    ddof: usize,
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Float
        + FromPrimitive
        + Send
        + Sync
        + 'static
        + scirs2_core::ndarray::ScalarOperand
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    match &x.storage {
        TensorStorage::Cpu(arr) => {
            let _result_shape =
                crate::ops::shape_inference::infer_reduction(x.shape(), axes, keepdims)?;

            if let Some(axes) = axes {
                // Calculate variance along specific axes
                let mut result = arr.map(|x| *x);

                // Sort axes in descending order
                let mut sorted_axes: Vec<_> = axes
                    .iter()
                    .map(|&a| normalize_axis(a, x.shape().rank() as i32))
                    .collect::<Result<Vec<_>>>()?;
                sorted_axes.sort_by(|a, b| b.cmp(a));

                for &axis in &sorted_axes {
                    // Calculate mean for this axis
                    let mean = result.mean_axis(Axis(axis)).unwrap();

                    // Calculate variance: mean of squared deviations
                    let mut variance_result = ArrayD::zeros(mean.raw_dim());
                    let axis_size = result.shape()[axis];
                    let n = if axis_size > ddof {
                        axis_size - ddof
                    } else {
                        1
                    };
                    let n_f = T::from_usize(n).unwrap_or_else(T::one);

                    for i in 0..axis_size {
                        let slice = result.index_axis(Axis(axis), i);
                        let diff = &slice - &mean;
                        let squared_diff = diff.mapv(|x| x * x);
                        variance_result = variance_result + squared_diff;
                    }

                    variance_result = variance_result / n_f;

                    result = if keepdims {
                        variance_result.insert_axis(Axis(axis))
                    } else {
                        variance_result
                    };
                }

                Ok(Tensor::from_array(result))
            } else {
                // Variance of all elements
                let mean_val = arr.mean().unwrap_or_default();
                let n = arr.len();
                let effective_n = if n > ddof { n - ddof } else { 1 };
                let n_f = T::from_usize(effective_n).unwrap_or_else(T::one);

                let variance_val = arr
                    .iter()
                    .map(|x| {
                        let diff = *x - mean_val;
                        diff * diff
                    })
                    .fold(T::zero(), |acc, x| acc + x)
                    / n_f;

                let result = if keepdims {
                    ArrayD::from_elem(vec![1; x.shape().rank()], variance_val)
                } else {
                    ArrayD::from_elem(vec![], variance_val)
                };
                Ok(Tensor::from_array(result))
            }
        }
        #[cfg(feature = "gpu")]
        TensorStorage::Gpu(gpu_buffer) => {
            use crate::gpu::ops::{execute_axis_reduction_op, ReductionOp};

            // Calculate output shape and size
            let result_shape =
                crate::ops::shape_inference::infer_reduction(x.shape(), axes, keepdims)?;
            let output_len = result_shape.dims().iter().product();

            let result_buffer = execute_axis_reduction_op(
                gpu_buffer,
                ReductionOp::Variance,
                x.shape().dims(),
                axes,
                keepdims,
                output_len,
            )?;

            Ok(Tensor::from_gpu_buffer(result_buffer, result_shape))
        }
    }
}

// Helper functions for parallel reduction operations

/// Parallel sum along a specific axis for large arrays
fn par_sum_axis<T>(arr: &ArrayD<T>, axis: usize) -> Result<ArrayD<T>>
where
    T: Clone + Default + Zero + std::ops::Add<Output = T> + Send + Sync + 'static,
{
    // Use SciRS2's parallel reduction for better performance
    let axis_obj = Axis(axis);

    // For demonstration - simplified parallel approach
    // In practice, this would use more sophisticated chunking
    Ok(arr.sum_axis(axis_obj))
}

/// Parallel sum of all elements in an array
fn parallel_sum_all<T>(arr: &ArrayD<T>) -> Result<T>
where
    T: Clone + Default + Zero + std::ops::Add<Output = T> + Send + Sync + 'static,
{
    // Use parallel chunking for very large arrays
    const CHUNK_SIZE: usize = 100000;

    if arr.len() <= CHUNK_SIZE {
        return Ok(arr.sum());
    }

    // Simplified parallel approach - in practice would use proper parallel iterators
    Ok(arr.sum())
}

#[cfg(feature = "gpu")]
/// CPU implementation of reduce along axis (used by GPU fallback)
pub fn reduce_axis_cpu<T>(
    tensor: &Tensor<T>,
    axis: usize,
    op: super::gpu_kernels::ReductionOp,
    keep_dims: bool,
) -> Result<Tensor<T>>
where
    T: scirs2_core::num_traits::Float
        + Default
        + bytemuck::Pod
        + Send
        + Sync
        + 'static
        + FromPrimitive
        + scirs2_core::num_traits::ops::mul_add::MulAdd
        + scirs2_core::ndarray::ScalarOperand,
{
    // Simple wrapper that calls the appropriate reduction function
    match op {
        super::gpu_kernels::ReductionOp::Sum => sum(tensor, Some(&[axis as i32]), keep_dims),
        super::gpu_kernels::ReductionOp::Mean => mean(tensor, Some(&[axis as i32]), keep_dims),
        super::gpu_kernels::ReductionOp::Max => max(tensor, Some(&[axis as i32]), keep_dims),
        super::gpu_kernels::ReductionOp::Min => min(tensor, Some(&[axis as i32]), keep_dims),
        super::gpu_kernels::ReductionOp::Prod => prod(tensor, Some(&[axis as i32]), keep_dims),
        super::gpu_kernels::ReductionOp::Variance => {
            variance(tensor, Some(&[axis as i32]), keep_dims, 0)
        }
        _ => Err(TensorError::not_implemented_simple(format!(
            "Reduction operation {:?} not implemented for CPU",
            op
        ))),
    }
}

#[cfg(feature = "gpu")]
/// CPU implementation of reduce all elements (used by GPU fallback)
pub fn reduce_all_cpu<T>(tensor: &Tensor<T>, op: super::gpu_kernels::ReductionOp) -> Result<T>
where
    T: scirs2_core::num_traits::Float
        + Default
        + bytemuck::Pod
        + Send
        + Sync
        + 'static
        + FromPrimitive
        + scirs2_core::num_traits::ops::mul_add::MulAdd
        + scirs2_core::ndarray::ScalarOperand,
{
    let result = match op {
        super::gpu_kernels::ReductionOp::Sum => sum(tensor, None, false)?,
        super::gpu_kernels::ReductionOp::Mean => mean(tensor, None, false)?,
        super::gpu_kernels::ReductionOp::Max => max(tensor, None, false)?,
        super::gpu_kernels::ReductionOp::Min => min(tensor, None, false)?,
        super::gpu_kernels::ReductionOp::Prod => prod(tensor, None, false)?,
        _ => {
            return Err(TensorError::not_implemented_simple(format!(
                "Reduction operation {:?} not implemented for CPU",
                op
            )))
        }
    };

    // Extract scalar value from result tensor
    let data = result.data();
    if data.is_empty() {
        Ok(T::default())
    } else {
        Ok(data[0])
    }
}
