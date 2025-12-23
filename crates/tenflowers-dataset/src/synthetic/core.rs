//! Core Types for Synthetic Dataset Generation
//!
//! This module contains the fundamental types and configuration structures
//! used across all synthetic dataset generation functions.

use crate::Dataset;
use tenflowers_core::{Result, Tensor, TensorError};

/// Configuration for synthetic dataset generation
#[derive(Debug, Clone)]
pub struct SyntheticConfig {
    pub n_samples: usize,
    pub random_seed: Option<u64>,
    pub noise_level: f64,
    pub shuffle: bool,
}

impl Default for SyntheticConfig {
    fn default() -> Self {
        Self {
            n_samples: 100,
            random_seed: None,
            noise_level: 0.1,
            shuffle: true,
        }
    }
}

impl SyntheticConfig {
    pub fn new(n_samples: usize) -> Self {
        Self {
            n_samples,
            ..Default::default()
        }
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.random_seed = Some(seed);
        self
    }

    pub fn with_noise(mut self, noise_level: f64) -> Self {
        self.noise_level = noise_level;
        self
    }

    pub fn with_shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }
}

/// A synthetic dataset containing features and labels
pub struct SyntheticDataset<T> {
    features: Tensor<T>,
    labels: Tensor<T>,
}

impl<T: Clone + Default + scirs2_core::numeric::Zero + Send + Sync + 'static> SyntheticDataset<T> {
    pub fn new(features: Tensor<T>, labels: Tensor<T>) -> Self {
        Self { features, labels }
    }
}

impl<T: Clone + Default + scirs2_core::numeric::Zero + Send + Sync + 'static> Dataset<T>
    for SyntheticDataset<T>
{
    fn len(&self) -> usize {
        self.features.shape().dims()[0]
    }

    fn get(&self, index: usize) -> Result<(Tensor<T>, Tensor<T>)> {
        if index >= self.len() {
            return Err(TensorError::invalid_argument(format!(
                "Index {} out of bounds for dataset of length {}",
                index,
                self.len()
            )));
        }

        // Get feature slice
        let feature_data = self.features.as_slice().ok_or_else(|| {
            TensorError::invalid_argument(
                "Cannot access tensor data (GPU tensor not supported)".to_string(),
            )
        })?;

        let label_data = self.labels.as_slice().ok_or_else(|| {
            TensorError::invalid_argument(
                "Cannot access tensor data (GPU tensor not supported)".to_string(),
            )
        })?;

        let feature_dim = self.features.shape().dims()[1];
        let label_dim = if self.labels.shape().rank() > 1 {
            self.labels.shape().dims()[1]
        } else {
            1
        };

        // Extract features for this sample
        let start_feat = index * feature_dim;
        let end_feat = start_feat + feature_dim;
        let sample_features = feature_data[start_feat..end_feat].to_vec();

        // Extract labels for this sample
        let sample_labels = if label_dim == 1 {
            vec![label_data[index].clone()]
        } else {
            let start_label = index * label_dim;
            let end_label = start_label + label_dim;
            label_data[start_label..end_label].to_vec()
        };

        let feature_shape = if feature_dim == 1 {
            vec![]
        } else {
            vec![feature_dim]
        };
        let label_shape = if label_dim == 1 {
            vec![]
        } else {
            vec![label_dim]
        };

        let feature_tensor = Tensor::from_vec(sample_features, &feature_shape)?;
        let label_tensor = Tensor::from_vec(sample_labels, &label_shape)?;

        Ok((feature_tensor, label_tensor))
    }
}

/// Main dataset generator structure
pub struct DatasetGenerator;
