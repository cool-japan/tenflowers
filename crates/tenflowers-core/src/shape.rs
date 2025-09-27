#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};
use std::ops::{Index, IndexMut};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    pub fn new(dims: Vec<usize>) -> Self {
        Self { dims }
    }

    pub fn from_slice(dims: &[usize]) -> Self {
        Self {
            dims: dims.to_vec(),
        }
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    pub fn len(&self) -> usize {
        self.dims.len()
    }

    pub fn is_empty(&self) -> bool {
        self.dims.is_empty()
    }

    pub fn size(&self) -> usize {
        self.dims.iter().product()
    }

    pub fn elements(&self) -> usize {
        self.size()
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub fn is_scalar(&self) -> bool {
        self.dims.is_empty()
    }

    pub fn is_compatible_with(&self, other: &Self) -> bool {
        if self.rank() != other.rank() {
            return false;
        }
        self.dims
            .iter()
            .zip(&other.dims)
            .all(|(a, b)| *a == *b || *a == 1 || *b == 1)
    }

    pub fn broadcast_shape(&self, other: &Self) -> Option<Self> {
        let rank = self.rank().max(other.rank());
        let mut result = vec![1; rank];

        for i in 0..self.rank() {
            result[rank - self.rank() + i] = self.dims[i];
        }

        for i in 0..other.rank() {
            let idx = rank - other.rank() + i;
            if result[idx] == 1 {
                result[idx] = other.dims[i];
            } else if other.dims[i] != 1 && result[idx] != other.dims[i] {
                return None;
            }
        }

        Some(Self::new(result))
    }

    /// Get an iterator over the dimensions
    pub fn iter(&self) -> std::slice::Iter<'_, usize> {
        self.dims.iter()
    }

    /// Convert dimensions to a vector
    pub fn to_vec(&self) -> Vec<usize> {
        self.dims.clone()
    }
}

impl Index<usize> for Shape {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.dims[index]
    }
}

impl IndexMut<usize> for Shape {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.dims[index]
    }
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for (i, dim) in self.dims.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{dim}")?;
        }
        write!(f, "]")
    }
}
