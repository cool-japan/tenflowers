//! Data transformation utilities for datasets
//!
//! This module provides common data preprocessing and augmentation transformations
//! that can be applied to datasets during training and inference.

use crate::Dataset;
use std::marker::PhantomData;
use tenflowers_core::{Result, Tensor};

// Re-export all transform modules
pub mod augmentation;
pub mod feature_engineering;
pub mod noise;
pub mod normalization;
pub mod pipeline;
pub mod profiling;
pub mod vision;

// Re-export commonly used types
pub use noise::*;
pub use normalization::*;

/// Trait for data transformations
pub trait Transform<T> {
    /// Apply the transformation to a sample
    fn apply(&self, sample: (Tensor<T>, Tensor<T>)) -> Result<(Tensor<T>, Tensor<T>)>;
}

/// Dataset wrapper that applies transformations to samples
pub struct TransformedDataset<T, D: Dataset<T>, Tr: Transform<T>> {
    dataset: D,
    transform: Tr,
    _phantom: PhantomData<T>,
}

impl<T, D: Dataset<T>, Tr: Transform<T>> TransformedDataset<T, D, Tr> {
    pub fn new(dataset: D, transform: Tr) -> Self {
        Self {
            dataset,
            transform,
            _phantom: PhantomData,
        }
    }
}

impl<T, D: Dataset<T>, Tr: Transform<T>> Dataset<T> for TransformedDataset<T, D, Tr> {
    fn len(&self) -> usize {
        self.dataset.len()
    }

    fn get(&self, index: usize) -> Result<(Tensor<T>, Tensor<T>)> {
        let sample = self.dataset.get(index)?;
        self.transform.apply(sample)
    }
}

/// Dataset extension trait for convenience methods
pub trait DatasetExt<T>: Dataset<T> + Sized {
    /// Apply a transform to this dataset
    fn transform<Tr: Transform<T>>(self, transform: Tr) -> TransformedDataset<T, Self, Tr> {
        TransformedDataset::new(self, transform)
    }
}

impl<T, D: Dataset<T>> DatasetExt<T> for D {}
