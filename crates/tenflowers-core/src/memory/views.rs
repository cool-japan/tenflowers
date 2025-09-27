//! Zero-copy tensor operations and memory views
//!
//! This module provides strided tensor views for efficient reshape and transpose
//! operations, along with memory aliasing detection for safe zero-copy operations.

use crate::{Result, TensorError};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Strided tensor view for zero-copy reshape and transpose operations
#[derive(Debug, Clone)]
pub struct StridedView {
    pub offset: usize,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub element_size: usize,
}

impl StridedView {
    /// Create a new strided view
    pub fn new(offset: usize, shape: Vec<usize>, strides: Vec<usize>, element_size: usize) -> Self {
        Self {
            offset,
            shape,
            strides,
            element_size,
        }
    }

    /// Create a strided view for transpose operation
    pub fn transpose(&self, axes: &[usize]) -> Result<StridedView> {
        if axes.len() != self.shape.len() {
            return Err(TensorError::invalid_argument(
                "Transpose axes must match tensor dimensions".to_string(),
            ));
        }

        let mut new_shape = Vec::new();
        let mut new_strides = Vec::new();

        for &axis in axes {
            if axis >= self.shape.len() {
                return Err(TensorError::invalid_argument(format!(
                    "Axis {} out of bounds for tensor with {} dimensions",
                    axis,
                    self.shape.len()
                )));
            }
            new_shape.push(self.shape[axis]);
            new_strides.push(self.strides[axis]);
        }

        Ok(StridedView {
            offset: self.offset,
            shape: new_shape,
            strides: new_strides,
            element_size: self.element_size,
        })
    }

    /// Create a strided view for reshape operation (zero-copy when possible)
    pub fn reshape(&self, new_shape: &[usize]) -> Result<StridedView> {
        // Check if reshape is possible without data copy
        let total_elements: usize = self.shape.iter().product();
        let new_total_elements: usize = new_shape.iter().product();

        if total_elements != new_total_elements {
            return Err(TensorError::invalid_argument(
                "Cannot reshape tensor: element count mismatch".to_string(),
            ));
        }

        // Check if tensor is contiguous
        if self.is_contiguous() {
            // Can reshape without copy
            let new_strides = compute_strides(new_shape, self.element_size);
            Ok(StridedView {
                offset: self.offset,
                shape: new_shape.to_vec(),
                strides: new_strides,
                element_size: self.element_size,
            })
        } else {
            // Non-contiguous tensor requires copy for reshape
            Err(TensorError::unsupported_operation_simple(
                "Reshape requires data copy for non-contiguous tensor".to_string(),
            ))
        }
    }

    /// Check if the tensor is contiguous in memory
    pub fn is_contiguous(&self) -> bool {
        let expected_strides = compute_strides(&self.shape, self.element_size);
        self.strides == expected_strides
    }

    /// Get the total size in bytes
    pub fn size_bytes(&self) -> usize {
        self.shape.iter().product::<usize>() * self.element_size
    }

    /// Create a slice view
    pub fn slice(&self, ranges: &[(usize, usize)]) -> Result<StridedView> {
        if ranges.len() != self.shape.len() {
            return Err(TensorError::invalid_argument(
                "Slice ranges must match tensor dimensions".to_string(),
            ));
        }

        let mut new_offset = self.offset;
        let mut new_shape = Vec::new();
        let mut new_strides = Vec::new();

        for (i, &(start, end)) in ranges.iter().enumerate() {
            if start >= end || end > self.shape[i] {
                return Err(TensorError::invalid_argument(format!(
                    "Invalid slice range [{}, {}) for dimension {} of size {}",
                    start, end, i, self.shape[i]
                )));
            }

            new_offset += start * self.strides[i];
            new_shape.push(end - start);
            new_strides.push(self.strides[i]);
        }

        Ok(StridedView {
            offset: new_offset,
            shape: new_shape,
            strides: new_strides,
            element_size: self.element_size,
        })
    }
}

/// Memory aliasing detector for safe zero-copy operations
#[derive(Debug)]
pub struct MemoryAliasDetector {
    #[allow(clippy::type_complexity)]
    active_views: Arc<Mutex<HashMap<usize, Vec<(usize, usize)>>>>, // buffer_id -> [(offset, size)]
}

impl Default for MemoryAliasDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryAliasDetector {
    /// Create a new memory alias detector
    pub fn new() -> Self {
        Self {
            active_views: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Check if a new view would create an alias
    pub fn check_alias(&self, buffer_id: usize, offset: usize, size: usize) -> bool {
        let active_views = self.active_views.lock().unwrap();

        if let Some(views) = active_views.get(&buffer_id) {
            for &(view_offset, view_size) in views {
                // Enhanced overlap detection: two ranges [a, b) and [c, d) overlap if max(a,c) < min(b,d)
                let start1 = offset;
                let end1 = offset + size;
                let start2 = view_offset;
                let end2 = view_offset + view_size;

                // Check for any overlap (including touching boundaries)
                if start1 < end2 && start2 < end1 {
                    return true; // Alias detected
                }
            }
        }

        false
    }

    /// Register a new view
    pub fn register_view(&self, buffer_id: usize, offset: usize, size: usize) {
        let mut active_views = self.active_views.lock().unwrap();
        active_views
            .entry(buffer_id)
            .or_default()
            .push((offset, size));
    }

    /// Unregister a view
    pub fn unregister_view(&self, buffer_id: usize, offset: usize, size: usize) {
        let mut active_views = self.active_views.lock().unwrap();
        if let Some(views) = active_views.get_mut(&buffer_id) {
            views.retain(|&(view_offset, view_size)| view_offset != offset || view_size != size);
            if views.is_empty() {
                active_views.remove(&buffer_id);
            }
        }
    }

    /// Get detailed information about potential aliases for a memory region
    pub fn get_alias_info(
        &self,
        buffer_id: usize,
        offset: usize,
        size: usize,
    ) -> Vec<(usize, usize, usize)> {
        let active_views = self.active_views.lock().unwrap();
        let mut aliases = Vec::new();

        if let Some(views) = active_views.get(&buffer_id) {
            for &(view_offset, view_size) in views {
                let start1 = offset;
                let end1 = offset + size;
                let start2 = view_offset;
                let end2 = view_offset + view_size;

                // Check for overlap and calculate overlap region
                if start1 < end2 && start2 < end1 {
                    let overlap_start = std::cmp::max(start1, start2);
                    let overlap_end = std::cmp::min(end1, end2);
                    let overlap_size = overlap_end - overlap_start;
                    aliases.push((overlap_start, overlap_size, view_size));
                }
            }
        }

        aliases
    }

    /// Check if a memory region would create partial aliases (useful for optimization decisions)
    pub fn check_partial_alias(&self, buffer_id: usize, offset: usize, size: usize) -> bool {
        let active_views = self.active_views.lock().unwrap();

        if let Some(views) = active_views.get(&buffer_id) {
            for &(view_offset, view_size) in views {
                let start1 = offset;
                let end1 = offset + size;
                let start2 = view_offset;
                let end2 = view_offset + view_size;

                // Check for partial overlap (not complete containment)
                if start1 < end2 && start2 < end1 {
                    // Check if it's not complete containment in either direction
                    let not_contained_in_existing = !(start1 >= start2 && end1 <= end2);
                    let not_containing_existing = !(start2 >= start1 && end2 <= end1);

                    // Only return true if NEITHER is completely contained (partial overlap)
                    if not_contained_in_existing && not_containing_existing {
                        return true;
                    }
                }
            }
        }

        false
    }

    /// Get statistics about active memory views
    pub fn get_alias_statistics(&self) -> (usize, usize) {
        let active_views = self.active_views.lock().unwrap();
        let total_buffers = active_views.len();
        let total_views: usize = active_views.values().map(|v| v.len()).sum();
        (total_buffers, total_views)
    }
}

/// Compute strides for a given shape
pub fn compute_strides(shape: &[usize], element_size: usize) -> Vec<usize> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut stride = element_size;

    for &dim in shape.iter().rev() {
        strides.push(stride);
        stride *= dim;
    }

    strides.reverse();
    strides
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strided_view_transpose() {
        let view = StridedView::new(0, vec![2, 3, 4], vec![48, 16, 4], 4);
        let transposed = view.transpose(&[2, 0, 1]).unwrap();

        assert_eq!(transposed.shape, vec![4, 2, 3]);
        assert_eq!(transposed.strides, vec![4, 48, 16]);
    }

    #[test]
    fn test_strided_view_reshape() {
        let view = StridedView::new(0, vec![2, 3, 4], vec![48, 16, 4], 4);
        let reshaped = view.reshape(&[6, 4]).unwrap();

        assert_eq!(reshaped.shape, vec![6, 4]);
        assert_eq!(reshaped.strides, vec![16, 4]);
    }

    #[test]
    fn test_strided_view_slice() {
        let view = StridedView::new(0, vec![4, 4], vec![16, 4], 4);
        let sliced = view.slice(&[(1, 3), (0, 2)]).unwrap();

        assert_eq!(sliced.shape, vec![2, 2]);
        assert_eq!(sliced.strides, vec![16, 4]);
        assert_eq!(sliced.offset, 16); // 1 * 16 + 0 * 4
    }

    #[test]
    fn test_memory_alias_detector() {
        let detector = MemoryAliasDetector::new();

        // Register a view
        detector.register_view(0, 0, 100);

        // Check for alias
        assert!(detector.check_alias(0, 50, 100)); // Overlaps
        assert!(!detector.check_alias(0, 100, 50)); // No overlap

        // Unregister
        detector.unregister_view(0, 0, 100);
        assert!(!detector.check_alias(0, 50, 100)); // No alias after unregister
    }

    #[test]
    fn test_compute_strides() {
        let strides = compute_strides(&[2, 3, 4], 4);
        assert_eq!(strides, vec![48, 16, 4]);
    }

    #[test]
    fn test_is_contiguous() {
        let contiguous_view = StridedView::new(0, vec![2, 3, 4], vec![48, 16, 4], 4);
        assert!(contiguous_view.is_contiguous());

        let non_contiguous_view = StridedView::new(0, vec![2, 3, 4], vec![32, 16, 4], 4);
        assert!(!non_contiguous_view.is_contiguous());
    }

    #[test]
    fn test_size_bytes() {
        let view = StridedView::new(0, vec![2, 3, 4], vec![48, 16, 4], 4);
        assert_eq!(view.size_bytes(), 96); // 2 * 3 * 4 * 4 = 96
    }

    #[test]
    fn test_invalid_transpose() {
        let view = StridedView::new(0, vec![2, 3], vec![12, 4], 4);

        // Wrong number of axes
        let result = view.transpose(&[1, 0, 2]);
        assert!(result.is_err());

        // Axis out of bounds
        let result = view.transpose(&[0, 3]);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_reshape() {
        let view = StridedView::new(0, vec![2, 3], vec![12, 4], 4);

        // Element count mismatch
        let result = view.reshape(&[2, 4]);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_slice() {
        let view = StridedView::new(0, vec![4, 4], vec![16, 4], 4);

        // Wrong number of dimensions
        let result = view.slice(&[(1, 3)]);
        assert!(result.is_err());

        // Invalid range
        let result = view.slice(&[(1, 1), (0, 2)]); // start >= end
        assert!(result.is_err());

        // Out of bounds
        let result = view.slice(&[(0, 5), (0, 2)]); // end > shape
        assert!(result.is_err());
    }

    #[test]
    fn test_alias_detection_edge_cases() {
        let detector = MemoryAliasDetector::new();

        // Test touching boundaries
        detector.register_view(0, 0, 100);
        assert!(!detector.check_alias(0, 100, 50)); // Adjacent, no overlap

        // Test complete containment
        detector.register_view(1, 10, 80);
        assert!(detector.check_alias(1, 20, 30)); // Contained within
        assert!(detector.check_alias(1, 0, 100)); // Contains the view

        // Test partial overlap
        assert!(detector.check_partial_alias(1, 50, 80)); // Partial overlap
        assert!(!detector.check_partial_alias(1, 15, 50)); // Complete containment
    }

    #[test]
    fn test_alias_info() {
        let detector = MemoryAliasDetector::new();
        detector.register_view(0, 10, 50);
        detector.register_view(0, 40, 30);

        let aliases = detector.get_alias_info(0, 35, 20);
        assert_eq!(aliases.len(), 2); // Overlaps with both views

        // Check the overlap details
        assert!(aliases
            .iter()
            .any(|&(start, size, _)| start == 40 && size == 15)); // Overlap with first view
        assert!(aliases
            .iter()
            .any(|&(start, size, _)| start == 40 && size == 15)); // Overlap with second view
    }

    #[test]
    fn test_alias_statistics() {
        let detector = MemoryAliasDetector::new();

        let (buffers, views) = detector.get_alias_statistics();
        assert_eq!(buffers, 0);
        assert_eq!(views, 0);

        detector.register_view(0, 0, 100);
        detector.register_view(0, 100, 100);
        detector.register_view(1, 0, 50);

        let (buffers, views) = detector.get_alias_statistics();
        assert_eq!(buffers, 2); // 2 different buffer IDs
        assert_eq!(views, 3); // 3 total views
    }
}
