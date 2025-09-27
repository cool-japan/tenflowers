use crate::{Result, TensorError};
use std::ops::Range;

/// Slice parameters with stride support
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SliceParams {
    pub start: Option<isize>,
    pub end: Option<isize>,
    pub step: Option<isize>,
}

impl SliceParams {
    /// Create a new slice parameters with default values
    pub fn new() -> Self {
        Self {
            start: None,
            end: None,
            step: Some(1),
        }
    }

    /// Create slice parameters with start, end, and step
    pub fn with_step(start: Option<isize>, end: Option<isize>, step: Option<isize>) -> Self {
        Self { start, end, step }
    }

    /// Convert to normalized start, end, step for a given dimension size
    pub fn normalize(&self, size: usize) -> Result<(usize, usize, isize)> {
        let size = size as isize;
        let step = self.step.unwrap_or(1);

        if step == 0 {
            return Err(TensorError::invalid_argument(
                "Slice step cannot be zero".to_string(),
            ));
        }

        let (start, end) = if step > 0 {
            let start = match self.start {
                Some(s) if s < 0 => (size + s).max(0) as usize,
                Some(s) => (s as usize).min(size as usize),
                None => 0,
            };
            let end = match self.end {
                Some(e) if e < 0 => (size + e).max(0) as usize,
                Some(e) => (e as usize).min(size as usize),
                None => size as usize,
            };
            (start, end)
        } else {
            let start = match self.start {
                Some(s) if s < 0 => (size + s).max(-1) as usize,
                Some(s) => (s as usize).min(size as usize - 1),
                None => size as usize - 1,
            };
            let end = match self.end {
                Some(e) if e < 0 => (size + e).max(-1) as usize,
                Some(e) => (e as usize).min(size as usize - 1),
                None => 0,
            };
            (start, end)
        };

        Ok((start, end, step))
    }
}

impl Default for SliceParams {
    fn default() -> Self {
        Self::new()
    }
}

impl From<Range<usize>> for SliceParams {
    fn from(range: Range<usize>) -> Self {
        Self {
            start: Some(range.start as isize),
            end: Some(range.end as isize),
            step: Some(1),
        }
    }
}

/// Strided tensor layout for efficient views and slicing
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StridedLayout {
    shape: Vec<usize>,
    strides: Vec<isize>,
    offset: usize,
}

impl StridedLayout {
    /// Create a new strided layout with default C-contiguous strides
    pub fn new(shape: Vec<usize>) -> Self {
        let strides = Self::compute_strides(&shape);
        Self {
            shape,
            strides,
            offset: 0,
        }
    }

    /// Create layout with custom strides
    pub fn with_strides(shape: Vec<usize>, strides: Vec<isize>, offset: usize) -> Result<Self> {
        if shape.len() != strides.len() {
            return Err(TensorError::invalid_argument(format!(
                "Shape and strides must have same length: {} != {}",
                strides.len(),
                shape.len()
            )));
        }

        Ok(Self {
            shape,
            strides,
            offset,
        })
    }

    /// Compute C-contiguous strides for a shape
    fn compute_strides(shape: &[usize]) -> Vec<isize> {
        let mut strides = vec![1isize; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1] as isize;
        }
        strides
    }

    /// Get the shape
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the strides
    pub fn strides(&self) -> &[isize] {
        &self.strides
    }

    /// Get the offset
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Get total number of elements
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Check if the layout is contiguous (C-order)
    pub fn is_contiguous(&self) -> bool {
        if self.offset != 0 {
            return false;
        }

        let expected_strides = Self::compute_strides(&self.shape);
        self.strides == expected_strides
    }

    /// Check if the layout is Fortran contiguous (F-order)
    pub fn is_fortran_contiguous(&self) -> bool {
        if self.offset != 0 {
            return false;
        }

        let mut expected_strides = vec![1isize; self.shape.len()];
        for i in 1..self.shape.len() {
            expected_strides[i] = expected_strides[i - 1] * self.shape[i - 1] as isize;
        }

        self.strides == expected_strides
    }

    /// Compute the linear index for a multi-dimensional index
    pub fn linear_index(&self, indices: &[usize]) -> Result<usize> {
        if indices.len() != self.shape.len() {
            return Err(TensorError::invalid_argument(format!(
                "Index dimension mismatch: {} != {}",
                indices.len(),
                self.shape.len()
            )));
        }

        let mut linear_idx = self.offset as isize;
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= self.shape[i] {
                return Err(TensorError::invalid_argument(format!(
                    "Index out of bounds: {} >= {}",
                    idx, self.shape[i]
                )));
            }
            linear_idx += idx as isize * self.strides[i];
        }

        Ok(linear_idx as usize)
    }

    /// Create a view by slicing along dimensions
    pub fn slice(&self, ranges: &[Range<usize>]) -> Result<Self> {
        if ranges.len() != self.shape.len() {
            return Err(TensorError::invalid_argument(format!(
                "Slice dimension mismatch: {} != {}",
                ranges.len(),
                self.shape.len()
            )));
        }

        let mut new_shape = Vec::with_capacity(self.shape.len());
        let mut new_offset = self.offset as isize;

        for (i, range) in ranges.iter().enumerate() {
            if range.start > range.end || range.end > self.shape[i] {
                return Err(TensorError::invalid_argument(format!(
                    "Invalid slice range {:?} for dimension size {}",
                    range, self.shape[i]
                )));
            }

            new_shape.push(range.end - range.start);
            new_offset += range.start as isize * self.strides[i];
        }

        if new_offset < 0 {
            return Err(TensorError::invalid_argument(
                "Slice operation resulted in negative offset".to_string(),
            ));
        }

        Ok(Self {
            shape: new_shape,
            strides: self.strides.clone(),
            offset: new_offset as usize,
        })
    }

    /// Create a view by slicing along dimensions with stride support
    pub fn slice_with_stride(&self, slice_params: &[SliceParams]) -> Result<Self> {
        if slice_params.len() != self.shape.len() {
            return Err(TensorError::invalid_argument(format!(
                "Slice dimension mismatch: {} != {}",
                slice_params.len(),
                self.shape.len()
            )));
        }

        let mut new_shape = Vec::with_capacity(self.shape.len());
        let mut new_strides = Vec::with_capacity(self.strides.len());
        let mut new_offset = self.offset as isize;

        for (i, slice_param) in slice_params.iter().enumerate() {
            let (start, end, step) = slice_param.normalize(self.shape[i])?;

            // Calculate the new dimension size
            let new_dim_size = if step > 0 {
                if start >= end {
                    0
                } else {
                    ((end - start) as isize + step - 1) / step
                }
            } else if start <= end {
                0
            } else {
                ((start as isize - end as isize) + (-step) - 1) / (-step)
            };

            new_shape.push(new_dim_size.max(0) as usize);
            new_strides.push(self.strides[i] * step);
            new_offset += start as isize * self.strides[i];
        }

        if new_offset < 0 {
            return Err(TensorError::invalid_argument(
                "Slice operation resulted in negative offset".to_string(),
            ));
        }

        Ok(Self {
            shape: new_shape,
            strides: new_strides,
            offset: new_offset as usize,
        })
    }

    /// Transpose the layout
    pub fn transpose(&self, axes: Option<&[usize]>) -> Result<Self> {
        let axes = if let Some(axes) = axes {
            if axes.len() != self.shape.len() {
                return Err(TensorError::invalid_argument(String::new()));
            }
            axes.to_vec()
        } else {
            // Default: reverse all axes
            (0..self.shape.len()).rev().collect()
        };

        // Check for valid permutation
        let mut seen = vec![false; self.shape.len()];
        for &ax in &axes {
            if ax >= self.shape.len() {
                return Err(TensorError::invalid_argument(String::new()));
            }
            if seen[ax] {
                return Err(TensorError::invalid_argument(String::new()));
            }
            seen[ax] = true;
        }

        let new_shape: Vec<_> = axes.iter().map(|&i| self.shape[i]).collect();
        let new_strides: Vec<_> = axes.iter().map(|&i| self.strides[i]).collect();

        Ok(Self {
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
        })
    }

    /// Reshape the layout (only works if contiguous)
    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Self> {
        if !self.is_contiguous() {
            return Err(TensorError::invalid_argument(String::new()));
        }

        let old_numel: usize = self.shape.iter().product();
        let new_numel: usize = new_shape.iter().product();

        if old_numel != new_numel {
            return Err(TensorError::invalid_argument(String::new()));
        }

        Ok(Self::new(new_shape))
    }

    /// Broadcast to a new shape
    pub fn broadcast_to(&self, target_shape: &[usize]) -> Result<Self> {
        // Check if we can broadcast
        if target_shape.len() < self.shape.len() {
            return Err(TensorError::invalid_argument(String::new()));
        }

        // Prepare new shape and strides
        let mut new_shape = vec![1; target_shape.len()];
        let mut new_strides = vec![0; target_shape.len()];
        let offset = target_shape.len() - self.shape.len();

        // Process existing dimensions
        for i in 0..self.shape.len() {
            let target_dim = target_shape[i + offset];
            let self_dim = self.shape[i];

            // Validate broadcast compatibility
            if self_dim != 1 && self_dim != target_dim {
                return Err(TensorError::invalid_argument(format!(
                    "Cannot broadcast dimension {self_dim} to {target_dim} at axis {i}"
                )));
            }

            new_shape[i + offset] = target_dim;
            new_strides[i + offset] = if self_dim == 1 { 0 } else { self.strides[i] };
        }

        // Set leading dimensions
        for i in 0..offset {
            new_shape[i] = target_shape[i];
            new_strides[i] = 0;
        }

        Ok(Self {
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
        })
    }

    /// Create an iterator over all valid indices
    pub fn indices_iter(&self) -> StridedIndicesIter {
        StridedIndicesIter::new(&self.shape)
    }
}

/// Iterator over multi-dimensional indices
pub struct StridedIndicesIter {
    shape: Vec<usize>,
    current: Vec<usize>,
    done: bool,
}

impl StridedIndicesIter {
    fn new(shape: &[usize]) -> Self {
        Self {
            shape: shape.to_vec(),
            current: vec![0; shape.len()],
            done: shape.contains(&0),
        }
    }
}

impl Iterator for StridedIndicesIter {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let result = self.current.clone();

        // Increment indices
        for i in (0..self.shape.len()).rev() {
            self.current[i] += 1;
            if self.current[i] < self.shape[i] {
                break;
            }
            if i == 0 {
                self.done = true;
            } else {
                self.current[i] = 0;
            }
        }

        Some(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strided_layout_basic() {
        let layout = StridedLayout::new(vec![2, 3, 4]);
        assert_eq!(layout.shape(), &[2, 3, 4]);
        assert_eq!(layout.strides(), &[12, 4, 1]);
        assert_eq!(layout.offset(), 0);
        assert!(layout.is_contiguous());
    }

    #[test]
    fn test_linear_index() {
        let layout = StridedLayout::new(vec![2, 3, 4]);
        assert_eq!(layout.linear_index(&[0, 0, 0]).unwrap(), 0);
        assert_eq!(layout.linear_index(&[1, 2, 3]).unwrap(), 23);
        assert_eq!(layout.linear_index(&[1, 0, 0]).unwrap(), 12);
    }

    #[test]
    fn test_slice() {
        let layout = StridedLayout::new(vec![4, 5, 6]);
        let sliced = layout.slice(&[1..3, 0..5, 2..4]).unwrap();
        assert_eq!(sliced.shape(), &[2, 5, 2]);
        assert_eq!(sliced.strides(), &[30, 6, 1]);
        assert_eq!(sliced.offset(), 32); // 1*30 + 0*6 + 2*1
    }

    #[test]
    fn test_transpose() {
        let layout = StridedLayout::new(vec![2, 3, 4]);
        let transposed = layout.transpose(Some(&[2, 0, 1])).unwrap();
        assert_eq!(transposed.shape(), &[4, 2, 3]);
        assert_eq!(transposed.strides(), &[1, 12, 4]);
    }

    #[test]
    fn test_broadcast() {
        let layout = StridedLayout::new(vec![1, 3, 1]);
        let broadcasted = layout.broadcast_to(&[2, 3, 4]).unwrap();
        assert_eq!(broadcasted.shape(), &[2, 3, 4]);
        assert_eq!(broadcasted.strides(), &[0, 1, 0]);
    }

    #[test]
    fn test_slice_params_normalize() {
        let params = SliceParams::with_step(Some(1), Some(4), Some(2));
        let (start, end, step) = params.normalize(6).unwrap();
        assert_eq!(start, 1);
        assert_eq!(end, 4);
        assert_eq!(step, 2);

        // Test negative indices
        let params = SliceParams::with_step(Some(-2), Some(-1), Some(1));
        let (start, end, step) = params.normalize(6).unwrap();
        assert_eq!(start, 4);
        assert_eq!(end, 5);
        assert_eq!(step, 1);
    }

    #[test]
    fn test_slice_with_stride() {
        let layout = StridedLayout::new(vec![6, 4]);

        // Test basic stride slicing - every 2nd element
        let slice_params = vec![
            SliceParams::with_step(Some(0), Some(6), Some(2)),
            SliceParams::with_step(Some(0), Some(4), Some(1)),
        ];
        let sliced = layout.slice_with_stride(&slice_params).unwrap();
        assert_eq!(sliced.shape(), &[3, 4]);
        assert_eq!(sliced.strides(), &[8, 1]); // stride doubled for dimension 0
        assert_eq!(sliced.offset(), 0);

        // Test negative step
        let slice_params = vec![
            SliceParams::with_step(Some(5), Some(0), Some(-2)),
            SliceParams::with_step(Some(0), Some(4), Some(1)),
        ];
        let sliced = layout.slice_with_stride(&slice_params).unwrap();
        assert_eq!(sliced.shape(), &[3, 4]);
        assert_eq!(sliced.strides(), &[-8, 1]); // negative stride for dimension 0
        assert_eq!(sliced.offset(), 20); // 5*4 + 0*1
    }

    #[test]
    fn test_slice_with_stride_default_params() {
        let layout = StridedLayout::new(vec![4, 4]);

        // Test with default parameters (equivalent to full slice)
        let slice_params = vec![SliceParams::default(), SliceParams::default()];
        let sliced = layout.slice_with_stride(&slice_params).unwrap();
        assert_eq!(sliced.shape(), &[4, 4]);
        assert_eq!(sliced.strides(), &[4, 1]);
        assert_eq!(sliced.offset(), 0);
    }

    #[test]
    fn test_slice_with_stride_from_range() {
        let layout = StridedLayout::new(vec![6, 4]);

        // Test converting from Range to SliceParams
        let slice_params = vec![SliceParams::from(1..5), SliceParams::from(0..4)];
        let sliced = layout.slice_with_stride(&slice_params).unwrap();
        assert_eq!(sliced.shape(), &[4, 4]);
        assert_eq!(sliced.strides(), &[4, 1]);
        assert_eq!(sliced.offset(), 4); // 1*4 + 0*1
    }
}
