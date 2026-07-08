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
                    result = result
                        .mean_axis(Axis(axis))
                        .expect("axis should be valid for mean reduction");
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
                    let mean = result
                        .mean_axis(Axis(axis))
                        .expect("axis should be valid for variance mean calculation");

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
        + scirs2_core::ndarray::ScalarOperand
        + scirs2_core::num_traits::Signed,
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
        super::gpu_kernels::ReductionOp::StdDev => {
            // StdDev = sqrt(Variance), mirrors the composition used in
            // gpu_kernels::gpu_reduce_axis for the GPU-resident case.
            let var = variance(tensor, Some(&[axis as i32]), keep_dims, 0)?;
            var.sqrt()
        }
        super::gpu_kernels::ReductionOp::L1Norm => {
            // L1 norm = sum of absolute values.
            let abs_t = tensor.abs()?;
            sum(&abs_t, Some(&[axis as i32]), keep_dims)
        }
        super::gpu_kernels::ReductionOp::L2Norm => {
            // L2 norm = sqrt(sum of squares).
            let sq = crate::ops::numpy_compat::square(tensor)?;
            let summed = sum(&sq, Some(&[axis as i32]), keep_dims)?;
            summed.sqrt()
        }
        // Any/All (boolean reductions) are intentionally left unimplemented here:
        // defining a "nonzero-as-true" convention for a generic float `T` is a
        // real design decision that this change does not want to guess at.
        // Open question for follow-up.
        super::gpu_kernels::ReductionOp::Any | super::gpu_kernels::ReductionOp::All => {
            Err(TensorError::not_implemented_simple(format!(
                "Reduction operation {:?} not implemented for CPU",
                op
            )))
        }
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
        + scirs2_core::ndarray::ScalarOperand
        + scirs2_core::num_traits::Signed,
{
    let result = match op {
        super::gpu_kernels::ReductionOp::Sum => sum(tensor, None, false)?,
        super::gpu_kernels::ReductionOp::Mean => mean(tensor, None, false)?,
        super::gpu_kernels::ReductionOp::Max => max(tensor, None, false)?,
        super::gpu_kernels::ReductionOp::Min => min(tensor, None, false)?,
        super::gpu_kernels::ReductionOp::Prod => prod(tensor, None, false)?,
        super::gpu_kernels::ReductionOp::StdDev => variance(tensor, None, false, 0)?.sqrt()?,
        super::gpu_kernels::ReductionOp::L1Norm => {
            let abs_t = tensor.abs()?;
            sum(&abs_t, None, false)?
        }
        super::gpu_kernels::ReductionOp::L2Norm => {
            let sq = crate::ops::numpy_compat::square(tensor)?;
            sum(&sq, None, false)?.sqrt()?
        }
        // Variance is intentionally NOT wired up here (pre-existing asymmetry vs.
        // reduce_axis_cpu, out of scope for this change). Any/All (boolean
        // reductions) are intentionally left unimplemented: defining a
        // "nonzero-as-true" convention for a generic float `T` is a real design
        // decision that this change does not want to guess at. Open question
        // for follow-up.
        super::gpu_kernels::ReductionOp::Variance
        | super::gpu_kernels::ReductionOp::Any
        | super::gpu_kernels::ReductionOp::All => {
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

#[cfg(feature = "gpu")]
#[cfg(test)]
mod tests {
    use super::super::gpu_kernels::ReductionOp;
    use super::*;

    /// Absolute tolerance used for float comparisons in this module's tests.
    const TOL: f32 = 1e-4;

    /// Fixed 2x3 f32 tensor shared by the tests below:
    /// ```text
    /// [[ 1.0, -2.0,  3.0],
    ///  [-4.0,  5.0, -6.0]]
    /// ```
    fn test_tensor() -> Tensor<f32> {
        Tensor::from_vec(vec![1.0f32, -2.0, 3.0, -4.0, 5.0, -6.0], &[2, 3])
            .expect("failed to build fixed 2x3 test tensor")
    }

    // ---- reduce_axis_cpu, axis = 1 (reduces each row of 3 elements to 1) ----
    //
    // row0 = [ 1, -2,  3]
    // row1 = [-4,  5, -6]

    #[test]
    fn test_reduce_axis_cpu_l1_norm() {
        let t = test_tensor();
        let result = reduce_axis_cpu(&t, 1, ReductionOp::L1Norm, false)
            .expect("L1Norm axis reduction should succeed");
        let data = result.data();
        // row0: |1| + |-2| + |3|  =  6
        // row1: |-4| + |5| + |-6| = 15
        assert_eq!(data.len(), 2);
        assert!(
            (data[0] - 6.0).abs() < TOL,
            "row0 L1 norm: got {}, expected 6.0",
            data[0]
        );
        assert!(
            (data[1] - 15.0).abs() < TOL,
            "row1 L1 norm: got {}, expected 15.0",
            data[1]
        );
    }

    #[test]
    fn test_reduce_axis_cpu_l2_norm() {
        let t = test_tensor();
        let result = reduce_axis_cpu(&t, 1, ReductionOp::L2Norm, false)
            .expect("L2Norm axis reduction should succeed");
        let data = result.data();
        // row0: sqrt(1^2 + 2^2 + 3^2) = sqrt(14) ~= 3.74166
        // row1: sqrt(4^2 + 5^2 + 6^2) = sqrt(77) ~= 8.77496
        assert_eq!(data.len(), 2);
        assert!(
            (data[0] - 3.74166).abs() < TOL,
            "row0 L2 norm: got {}, expected sqrt(14) ~= 3.74166",
            data[0]
        );
        assert!(
            (data[1] - 8.77496).abs() < TOL,
            "row1 L2 norm: got {}, expected sqrt(77) ~= 8.77496",
            data[1]
        );
    }

    #[test]
    fn test_reduce_axis_cpu_std_dev() {
        let t = test_tensor();
        let result = reduce_axis_cpu(&t, 1, ReductionOp::StdDev, false)
            .expect("StdDev axis reduction should succeed");
        let data = result.data();
        // row0: mean = 2/3;  population variance (ddof=0) = 38/9 ~= 4.22222
        //       stddev = sqrt(4.22222) ~= 2.05480
        // row1: mean = -5/3; population variance (ddof=0) = 206/9 ~= 22.88889
        //       stddev = sqrt(22.88889) ~= 4.78423
        assert_eq!(data.len(), 2);
        assert!(
            (data[0] - 2.05480).abs() < TOL,
            "row0 stddev: got {}, expected ~= 2.05480",
            data[0]
        );
        assert!(
            (data[1] - 4.78423).abs() < TOL,
            "row1 stddev: got {}, expected ~= 4.78423",
            data[1]
        );
    }

    // ---- reduce_all_cpu (whole-tensor scalar reduction over all 6 elements) ----

    #[test]
    fn test_reduce_all_cpu_l1_norm() {
        let t = test_tensor();
        let result =
            reduce_all_cpu(&t, ReductionOp::L1Norm).expect("L1Norm full reduction should succeed");
        // |1| + |-2| + |3| + |-4| + |5| + |-6| = 1+2+3+4+5+6 = 21
        assert!(
            (result - 21.0).abs() < TOL,
            "whole-tensor L1 norm: got {}, expected 21.0",
            result
        );
    }

    #[test]
    fn test_reduce_all_cpu_l2_norm() {
        let t = test_tensor();
        let result =
            reduce_all_cpu(&t, ReductionOp::L2Norm).expect("L2Norm full reduction should succeed");
        // sqrt(1+4+9+16+25+36) = sqrt(91) ~= 9.53939
        assert!(
            (result - 9.53939).abs() < TOL,
            "whole-tensor L2 norm: got {}, expected sqrt(91) ~= 9.53939",
            result
        );
    }

    #[test]
    fn test_reduce_all_cpu_std_dev() {
        let t = test_tensor();
        let result =
            reduce_all_cpu(&t, ReductionOp::StdDev).expect("StdDev full reduction should succeed");
        // mean = -3/6 = -0.5; population variance (ddof=0) = 89.5/6 ~= 14.91667
        // stddev = sqrt(14.91667) ~= 3.86221
        assert!(
            (result - 3.86221).abs() < TOL,
            "whole-tensor stddev: got {}, expected ~= 3.86221",
            result
        );
    }

    // ---- Any/All: intentionally unimplemented (open design question) ----
    //
    // These lock in current, deliberate behavior: a boolean "nonzero-as-true"
    // reduction convention for a generic float `T` needs a real design
    // decision that is out of scope here, so both ops must keep erroring
    // instead of silently guessing at semantics.

    #[test]
    fn test_reduce_axis_cpu_any_all_not_implemented() {
        let t = test_tensor();
        assert!(reduce_axis_cpu(&t, 1, ReductionOp::Any, false).is_err());
        assert!(reduce_axis_cpu(&t, 1, ReductionOp::All, false).is_err());
    }

    #[test]
    fn test_reduce_all_cpu_variance_any_all_not_implemented() {
        let t = test_tensor();
        // Variance is intentionally not wired up in reduce_all_cpu (pre-existing
        // asymmetry vs. reduce_axis_cpu; out of scope for this change).
        assert!(reduce_all_cpu(&t, ReductionOp::Variance).is_err());
        assert!(reduce_all_cpu(&t, ReductionOp::Any).is_err());
        assert!(reduce_all_cpu(&t, ReductionOp::All).is_err());
    }
}
