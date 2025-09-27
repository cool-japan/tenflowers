//! Utility functions for gradient operations
//!
//! This module contains common helper functions used throughout the gradient operations,
//! including broadcasting utilities, index normalization, and tensor manipulation helpers.

use num_traits::{One, Zero};
use tenflowers_core::ops::broadcast_to;
use tenflowers_core::{Result, Shape, Tensor, TensorError};

/// Helper function to "unbroadcast" a gradient tensor
/// This sums over dimensions that were broadcasted during the forward pass
pub fn unbroadcast<T>(grad: &Tensor<T>, target_shape: &Shape) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + std::ops::Add<Output = T>
        + Send
        + Sync
        + 'static
        + num_traits::One
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    let grad_shape = grad.shape().dims();
    let target_dims = target_shape.dims();

    // If shapes match, no unbroadcasting needed
    if grad_shape == target_dims {
        return Ok(grad.clone());
    }

    let mut result = grad.clone();

    // Sum over dimensions that need to be reduced
    let grad_ndim = grad_shape.len();
    let target_ndim = target_dims.len();

    // Handle dimensions that need to be summed out completely
    // (when grad has more dimensions than target)
    if grad_ndim > target_ndim {
        for _ in 0..(grad_ndim - target_ndim) {
            result = result.sum(Some(&[0]), false)?;
        }
    }

    // Handle dimensions that were broadcasted from size 1
    for i in 0..target_ndim {
        let current_shape = result.shape().dims();
        let current_dim = current_shape[i];
        let target_dim = target_dims[i];

        if current_dim != target_dim {
            if target_dim == 1 {
                // Sum along this dimension and keep dimension
                result = result.sum(Some(&[i as i32]), true)?;
            } else if current_dim != 1 {
                return Err(TensorError::shape_mismatch(
                    "unbroadcast",
                    &format!("{target_dims:?}"),
                    &format!("{grad_shape:?}"),
                ));
            }
        }
    }

    // Reshape to ensure correct output shape
    if result.shape().dims() != target_dims {
        result = result.reshape(target_dims)?;
    }

    Ok(result)
}

/// Helper function to normalize negative indices
pub fn normalize_index(index: isize, size: usize) -> Result<usize> {
    if index < 0 {
        let positive_index = size as isize + index;
        if positive_index < 0 {
            Err(TensorError::InvalidArgument {
                operation: "index_normalization".to_string(),
                reason: format!("Index {index} is out of bounds for size {size}"),
                context: None,
            })
        } else {
            Ok(positive_index as usize)
        }
    } else {
        let idx = index as usize;
        if idx > size {
            Ok(size) // Clamp to size
        } else {
            Ok(idx)
        }
    }
}

/// Helper function to expand tensors for reduction operations
pub fn expand_for_reduction<T>(
    tensor: &Tensor<T>,
    target_shape: &[usize],
    axes: Option<&[i32]>,
) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + One + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    if let Some(axes) = axes {
        // Add dimensions back that were reduced
        let mut expanded_shape = vec![1; target_shape.len()];
        let mut _tensor_dim_idx = 0;

        for (i, &dim_size) in target_shape.iter().enumerate() {
            let axis_reduced = axes.iter().any(|&axis| {
                let normalized_axis = if axis < 0 {
                    (target_shape.len() as i32 + axis) as usize
                } else {
                    axis as usize
                };
                normalized_axis == i
            });

            if !axis_reduced {
                expanded_shape[i] = dim_size;
            }
        }

        let expanded = tensor.reshape(&expanded_shape)?;
        broadcast_to(&expanded, target_shape)
    } else {
        // No specific axes, just broadcast to target shape
        broadcast_to(tensor, target_shape)
    }
}

/// Helper function to create a mask for maximum elements
pub fn create_max_mask<T>(
    input: &Tensor<T>,
    max_values: &Tensor<T>,
    axes: Option<&[i32]>,
    keepdims: bool,
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + One
        + PartialOrd
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // Broadcast max_values to input shape for comparison
    let broadcasted_max = if axes.is_some() && !keepdims {
        let expanded_max = expand_for_reduction(max_values, input.shape().dims(), axes)?;
        expanded_max
    } else {
        broadcast_to(max_values, input.shape().dims())?
    };

    // Create mask where input equals max values and convert bool to T
    let bool_mask = input.eq(&broadcasted_max)?;
    // Convert bool tensor to T tensor (1 for true, 0 for false)
    let zero_tensor = Tensor::zeros(input.shape().dims());
    let one_tensor = Tensor::ones(input.shape().dims());
    tenflowers_core::ops::where_op(&bool_mask, &one_tensor, &zero_tensor)
}

/// Helper function to create a mask for minimum elements
pub fn create_min_mask<T>(
    input: &Tensor<T>,
    min_values: &Tensor<T>,
    axes: Option<&[i32]>,
    keepdims: bool,
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + One
        + PartialOrd
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // Broadcast min_values to input shape for comparison
    let broadcasted_min = if axes.is_some() && !keepdims {
        let expanded_min = expand_for_reduction(min_values, input.shape().dims(), axes)?;
        expanded_min
    } else {
        broadcast_to(min_values, input.shape().dims())?
    };

    // Create mask where input equals min values and convert bool to T
    let bool_mask = input.eq(&broadcasted_min)?;
    // Convert bool tensor to T tensor (1 for true, 0 for false)
    let zero_tensor = Tensor::zeros(input.shape().dims());
    let one_tensor = Tensor::ones(input.shape().dims());
    tenflowers_core::ops::where_op(&bool_mask, &one_tensor, &zero_tensor)
}

/// Helper function to calculate batch size for BatchNorm
pub fn calculate_batch_size(shape: &[usize]) -> usize {
    if shape.is_empty() {
        1
    } else {
        shape[0]
    }
}

/// Helper function to get an element from a 4D tensor at position [b, c, h, w]
pub fn get_4d_element<T>(tensor: &Tensor<T>, b: usize, c: usize, h: usize, w: usize) -> Result<T>
where
    T: Clone + Default,
{
    let shape = tensor.shape().dims();
    if shape.len() != 4 {
        return Err(TensorError::invalid_shape_simple(
            "Expected 4D tensor".to_string(),
        ));
    }

    let flat_index =
        b * shape[1] * shape[2] * shape[3] + c * shape[2] * shape[3] + h * shape[3] + w;

    if let Some(data) = tensor.as_slice() {
        if flat_index < data.len() {
            Ok(data[flat_index].clone())
        } else {
            Err(TensorError::invalid_shape_simple(
                "Index out of bounds".to_string(),
            ))
        }
    } else {
        Err(TensorError::invalid_shape_simple(
            "Cannot access tensor data".to_string(),
        ))
    }
}

/// Helper function to get element from 5D tensor
pub fn get_5d_element<T>(
    tensor: &Tensor<T>,
    b: usize,
    c: usize,
    d: usize,
    h: usize,
    w: usize,
) -> Result<T>
where
    T: Clone + Default,
{
    let shape = tensor.shape().dims();
    if shape.len() != 5 {
        return Err(TensorError::invalid_shape_simple(
            "Expected 5D tensor".to_string(),
        ));
    }

    let flat_index = b * shape[1] * shape[2] * shape[3] * shape[4]
        + c * shape[2] * shape[3] * shape[4]
        + d * shape[3] * shape[4]
        + h * shape[4]
        + w;

    if let Some(data) = tensor.as_slice() {
        if flat_index < data.len() {
            Ok(data[flat_index].clone())
        } else {
            Err(TensorError::invalid_shape_simple(
                "Index out of bounds".to_string(),
            ))
        }
    } else {
        Err(TensorError::invalid_shape_simple(
            "Cannot access tensor data".to_string(),
        ))
    }
}
