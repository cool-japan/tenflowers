//! Common Helper Functions and GPU Dispatch Utilities
//!
//! This module contains shared helper functions and GPU dispatch utilities
//! used across different tensor manipulation operations.

#[cfg(feature = "gpu")]
use crate::gpu::buffer::GpuBuffer;

/// Calculate strides for a given shape
pub fn calculate_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Convert flat index to multi-dimensional coordinates
pub fn flat_to_coords(flat_idx: usize, shape: &[usize]) -> Vec<usize> {
    let mut coords = vec![0; shape.len()];
    let mut remaining = flat_idx;

    let strides = calculate_strides(shape);
    for (i, &stride) in strides.iter().enumerate() {
        coords[i] = remaining / stride;
        remaining %= stride;
    }

    coords
}

/// Convert multi-dimensional coordinates to flat index
pub fn coords_to_flat(coords: &[usize], strides: &[usize]) -> usize {
    coords.iter().zip(strides.iter()).map(|(c, s)| c * s).sum()
}

/// Helper function to calculate broadcast indices
pub fn broadcast_indices(
    out_indices: &[usize],
    in_shape: &[usize],
    out_shape: &[usize],
) -> Vec<usize> {
    let rank_diff = out_shape.len() - in_shape.len();
    out_indices
        .iter()
        .skip(rank_diff)
        .zip(in_shape)
        .map(|(&out_idx, &in_dim)| if in_dim == 1 { 0 } else { out_idx })
        .collect()
}

// GPU dispatch functions are included in their respective operation modules
// to keep code organization clean and focused
