//! Tensor Indexing Operations
//!
//! This module provides direct indexing capabilities for tensors,
//! allowing for convenient access to tensor elements using
//! standard Rust indexing syntax.

use super::core::{Tensor, TensorStorage};
use std::ops::Index;

// Index trait implementation for tensor[indices] syntax
impl<T: Clone> Index<&[usize]> for Tensor<T> {
    type Output = T;

    fn index(&self, index: &[usize]) -> &Self::Output {
        match &self.storage {
            TensorStorage::Cpu(arr) => {
                if index.len() != arr.ndim() {
                    panic!(
                        "Index dimension mismatch: expected {} dimensions, got {}",
                        arr.ndim(),
                        index.len()
                    );
                }
                arr.get(index).expect("Index out of bounds")
            }
            #[cfg(feature = "gpu")]
            TensorStorage::Gpu(_) => {
                panic!("Direct indexing not supported for GPU tensors. Use .get() or convert to CPU first.")
            }
        }
    }
}

// Index trait implementation for single-dimension indexing
impl<T: Clone> Index<usize> for Tensor<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        match &self.storage {
            TensorStorage::Cpu(arr) => {
                if arr.ndim() != 1 {
                    panic!(
                        "Single index only supported for 1D tensors, but tensor has {} dimensions",
                        arr.ndim()
                    );
                }
                arr.get([index]).expect("Index out of bounds")
            }
            #[cfg(feature = "gpu")]
            TensorStorage::Gpu(_) => {
                panic!("Direct indexing not supported for GPU tensors. Use .get() or convert to CPU first.")
            }
        }
    }
}
