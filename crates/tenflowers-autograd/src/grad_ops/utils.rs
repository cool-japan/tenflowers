//! Utility functions for gradient computations
//!
//! This module provides helper functions that are commonly used across
//! different gradient computation operations, such as unbroadcasting
//! and shape handling utilities.

use num_traits::Zero;
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

/// Represents a slice specification for a single dimension
#[derive(Debug, Clone)]
pub struct SliceSpec {
    pub start: Option<isize>,
    pub end: Option<isize>,
    pub step: Option<isize>,
}

impl SliceSpec {
    /// Create a new slice specification
    pub fn new(start: Option<isize>, end: Option<isize>, step: Option<isize>) -> Self {
        Self { start, end, step }
    }

    /// Create a slice specification for all elements in a dimension
    pub fn all() -> Self {
        Self {
            start: None,
            end: None,
            step: Some(1),
        }
    }

    /// Create a slice specification for a single index
    pub fn single(index: isize) -> Self {
        Self {
            start: Some(index),
            end: Some(index + 1),
            step: Some(1),
        }
    }

    /// Create a slice specification for a range
    pub fn range(start: isize, end: isize) -> Self {
        Self {
            start: Some(start),
            end: Some(end),
            step: Some(1),
        }
    }

    /// Create a slice specification with a step
    pub fn range_with_step(start: isize, end: isize, step: isize) -> Self {
        Self {
            start: Some(start),
            end: Some(end),
            step: Some(step),
        }
    }
}

/// Compute the effective slice indices for a given dimension size
pub fn compute_slice_indices(spec: &SliceSpec, dim_size: usize) -> (usize, usize, usize) {
    let size = dim_size as isize;

    // Handle start index
    let start = match spec.start {
        Some(s) if s < 0 => (size + s).max(0) as usize,
        Some(s) => (s.min(size).max(0)) as usize,
        None => 0,
    };

    // Handle end index
    let end = match spec.end {
        Some(e) if e < 0 => (size + e).max(0) as usize,
        Some(e) => (e.min(size).max(0)) as usize,
        None => dim_size,
    };

    // Handle step
    let step = spec.step.unwrap_or(1).unsigned_abs();
    let step = if step == 0 { 1 } else { step };

    (start, end, step)
}

/// Helper function to create a zeros tensor with the same device and dtype as reference
pub fn zeros_like<T>(reference: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero,
{
    Ok(Tensor::zeros(reference.shape().dims()))
}

/// Helper function to create a ones tensor with the same device and dtype as reference
pub fn ones_like<T>(reference: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + num_traits::One,
{
    Ok(Tensor::ones(reference.shape().dims()))
}

/// Utility to safely compute element count from shape
pub fn element_count(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// Utility to validate that two tensors have compatible shapes for broadcasting
pub fn check_broadcast_compatible(shape1: &[usize], shape2: &[usize]) -> Result<Vec<usize>> {
    let mut result_shape = Vec::new();
    let ndim1 = shape1.len();
    let ndim2 = shape2.len();
    let max_ndim = ndim1.max(ndim2);

    for i in 0..max_ndim {
        let dim1 = if i < ndim1 { shape1[ndim1 - 1 - i] } else { 1 };
        let dim2 = if i < ndim2 { shape2[ndim2 - 1 - i] } else { 1 };

        let result_dim = if dim1 == 1 {
            dim2
        } else if dim2 == 1 || dim1 == dim2 {
            dim1
        } else {
            return Err(TensorError::shape_mismatch(
                "broadcast",
                &format!("{shape1:?}"),
                &format!("{shape2:?}"),
            ));
        };

        result_shape.push(result_dim);
    }

    result_shape.reverse();
    Ok(result_shape)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenflowers_core::Tensor;

    #[test]
    fn test_slice_spec_creation() {
        let spec1 = SliceSpec::all();
        assert_eq!(spec1.start, None);
        assert_eq!(spec1.end, None);
        assert_eq!(spec1.step, Some(1));

        let spec2 = SliceSpec::single(5);
        assert_eq!(spec2.start, Some(5));
        assert_eq!(spec2.end, Some(6));
        assert_eq!(spec2.step, Some(1));

        let spec3 = SliceSpec::range(2, 8);
        assert_eq!(spec3.start, Some(2));
        assert_eq!(spec3.end, Some(8));
        assert_eq!(spec3.step, Some(1));
    }

    #[test]
    fn test_compute_slice_indices() {
        let spec = SliceSpec::range(2, 8);
        let (start, end, step) = compute_slice_indices(&spec, 10);
        assert_eq!(start, 2);
        assert_eq!(end, 8);
        assert_eq!(step, 1);

        // Test negative indices
        let spec_neg = SliceSpec::range(-3, -1);
        let (start, end, step) = compute_slice_indices(&spec_neg, 10);
        assert_eq!(start, 7);
        assert_eq!(end, 9);
        assert_eq!(step, 1);
    }

    #[test]
    fn test_check_broadcast_compatible() {
        // Compatible shapes
        let result = check_broadcast_compatible(&[3, 1, 5], &[1, 4, 5]).unwrap();
        assert_eq!(result, vec![3, 4, 5]);

        // Same shapes
        let result = check_broadcast_compatible(&[2, 3], &[2, 3]).unwrap();
        assert_eq!(result, vec![2, 3]);

        // Incompatible shapes should fail
        let result = check_broadcast_compatible(&[3, 4], &[2, 5]);
        assert!(result.is_err());
    }

    #[test]
    fn test_element_count() {
        assert_eq!(element_count(&[2, 3, 4]), 24);
        assert_eq!(element_count(&[1]), 1);
        assert_eq!(element_count(&[]), 1);
    }
}
