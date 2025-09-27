//! Common helper functions for reduction operations
//!
//! This module contains shared utility functions used across different reduction operations
//! to avoid code duplication and maintain consistency.

use crate::{Result, TensorError};

/// Normalize negative axis indices
///
/// Converts negative axis indices to positive ones and validates that the axis
/// is within the valid range for a tensor of the given rank.
///
/// # Arguments
/// * `axis` - The axis index (can be negative)
/// * `rank` - The rank (number of dimensions) of the tensor
///
/// # Returns
/// * `Ok(usize)` - The normalized positive axis index
/// * `Err(TensorError)` - If the axis is out of range
///
/// # Examples
/// ```
/// use tenflowers_core::ops::reduction::normalize_axis;
/// // For a 3D tensor (rank = 3):
/// assert_eq!(normalize_axis(0, 3).unwrap(), 0);  // First axis
/// assert_eq!(normalize_axis(-1, 3).unwrap(), 2); // Last axis
/// assert_eq!(normalize_axis(-3, 3).unwrap(), 0); // First axis (negative)
///
/// // Out of range examples:
/// assert!(normalize_axis(3, 3).is_err());  // Too large
/// assert!(normalize_axis(-4, 3).is_err()); // Too negative
/// ```
pub fn normalize_axis(axis: i32, rank: i32) -> Result<usize> {
    let normalized = if axis < 0 { rank + axis } else { axis };
    if normalized < 0 || normalized >= rank {
        Err(TensorError::invalid_argument(format!(
            "Axis {axis} out of range for tensor of rank {rank}"
        )))
    } else {
        Ok(normalized as usize)
    }
}
