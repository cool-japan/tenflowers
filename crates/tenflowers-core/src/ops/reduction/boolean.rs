//! Boolean reduction operations for tensors.
//!
//! This module contains boolean reduction operations that check conditions
//! across tensor elements along specified axes.

use super::common::normalize_axis;
use crate::tensor::TensorStorage;
use crate::{Result, Tensor, TensorError};
use scirs2_core::ndarray::{ArrayD, Axis};

/// Check if all elements are true along specified axes.
///
/// # Arguments
///
/// * `x` - Input boolean tensor
/// * `axes` - Optional axes to reduce along. If None, reduces all axes.
/// * `keepdims` - Whether to keep reduced dimensions with size 1
///
/// # Returns
///
/// Boolean tensor with the reduction result
pub fn all(x: &Tensor<bool>, axes: Option<&[i32]>, keepdims: bool) -> Result<Tensor<bool>> {
    match &x.storage {
        TensorStorage::Cpu(arr) => {
            let _result_shape =
                crate::ops::shape_inference::infer_reduction(x.shape(), axes, keepdims)?;

            if let Some(axes) = axes {
                let mut result = arr.clone();

                let mut sorted_axes: Vec<_> = axes
                    .iter()
                    .map(|&a| normalize_axis(a, x.shape().rank() as i32))
                    .collect::<Result<Vec<_>>>()?;
                sorted_axes.sort_by(|a, b| b.cmp(a));

                for &axis in &sorted_axes {
                    result = result.map_axis(Axis(axis), |view| view.iter().all(|&x| x));
                    if keepdims {
                        result = result.insert_axis(Axis(axis));
                    }
                }

                Ok(Tensor::from_array(result))
            } else {
                let all_val = arr.iter().all(|&x| x);
                let result = if keepdims {
                    ArrayD::from_elem(vec![1; x.shape().rank()], all_val)
                } else {
                    ArrayD::from_elem(vec![], all_val)
                };
                Ok(Tensor::from_array(result))
            }
        }
        #[cfg(feature = "gpu")]
        TensorStorage::Gpu(_gpu_buffer) => {
            // Boolean operations don't support GPU since bool doesn't implement Pod
            // This case should not normally occur since bool tensors can't be created on GPU
            Err(TensorError::unsupported_operation_simple(
                "Boolean operations not supported on GPU - bool type doesn't implement Pod trait"
                    .to_string(),
            ))
        }
    }
}

/// Check if any elements are true along specified axes.
///
/// # Arguments
///
/// * `x` - Input boolean tensor
/// * `axes` - Optional axes to reduce along. If None, reduces all axes.
/// * `keepdims` - Whether to keep reduced dimensions with size 1
///
/// # Returns
///
/// Boolean tensor with the reduction result
pub fn any(x: &Tensor<bool>, axes: Option<&[i32]>, keepdims: bool) -> Result<Tensor<bool>> {
    match &x.storage {
        TensorStorage::Cpu(arr) => {
            let _result_shape =
                crate::ops::shape_inference::infer_reduction(x.shape(), axes, keepdims)?;

            if let Some(axes) = axes {
                let mut result = arr.clone();

                let mut sorted_axes: Vec<_> = axes
                    .iter()
                    .map(|&a| normalize_axis(a, x.shape().rank() as i32))
                    .collect::<Result<Vec<_>>>()?;
                sorted_axes.sort_by(|a, b| b.cmp(a));

                for &axis in &sorted_axes {
                    result = result.map_axis(Axis(axis), |view| view.iter().any(|&x| x));
                    if keepdims {
                        result = result.insert_axis(Axis(axis));
                    }
                }

                Ok(Tensor::from_array(result))
            } else {
                let any_val = arr.iter().any(|&x| x);
                let result = if keepdims {
                    ArrayD::from_elem(vec![1; x.shape().rank()], any_val)
                } else {
                    ArrayD::from_elem(vec![], any_val)
                };
                Ok(Tensor::from_array(result))
            }
        }
        #[cfg(feature = "gpu")]
        TensorStorage::Gpu(_gpu_buffer) => {
            // Boolean operations don't support GPU since bool doesn't implement Pod
            // This case should not normally occur since bool tensors can't be created on GPU
            Err(TensorError::unsupported_operation_simple(
                "Boolean operations not supported on GPU - bool type doesn't implement Pod trait"
                    .to_string(),
            ))
        }
    }
}
