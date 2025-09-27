//! Statistics computation implementation
//!
//! This module contains the core logic for computing dataset statistics
//! including mean, standard deviation, histograms, and min/max values.

use crate::Dataset;
use std::collections::HashMap;
use tenflowers_core::{Result, Tensor, TensorError};

use super::core::{DatasetStats, Histogram, StatisticsConfig};

/// Statistics computer for datasets
pub struct DatasetStatisticsComputer;

impl DatasetStatisticsComputer {
    /// Compute statistics for a dataset
    pub fn compute<T, D>(dataset: &D, config: StatisticsConfig) -> Result<DatasetStats<T>>
    where
        T: Clone
            + Default
            + num_traits::Zero
            + num_traits::Float
            + std::fmt::Debug
            + Send
            + Sync
            + 'static,
        D: Dataset<T>,
    {
        if dataset.is_empty() {
            return Err(TensorError::invalid_argument(
                "Cannot compute statistics on empty dataset".to_string(),
            ));
        }

        let sample_count = dataset.len();
        let first_sample = dataset.get(0)?;
        let feature_count = first_sample.0.shape().dims().iter().product::<usize>();

        let mut stats = DatasetStats::new(feature_count, sample_count);

        // Collect all feature vectors
        let mut all_features = Vec::new();
        for i in 0..sample_count {
            let (features, _) = dataset.get(i)?;
            let feature_vec = Self::tensor_to_vec(&features)?;
            all_features.push(feature_vec);
        }

        // Compute mean
        if config.compute_mean {
            stats.mean = Some(Self::compute_mean(&all_features)?);
        }

        // Compute standard deviation
        if config.compute_std {
            let mean = if let Some(ref mean) = stats.mean {
                mean.clone()
            } else {
                Self::compute_mean(&all_features)?
            };
            stats.std = Some(Self::compute_std(&all_features, &mean)?);
        }

        // Compute min/max
        if config.compute_min_max {
            let (min, max) = Self::compute_min_max(&all_features)?;
            stats.min = Some(min);
            stats.max = Some(max);
        }

        // Compute histogram
        if config.compute_histogram {
            let min = if let Some(ref min) = stats.min {
                min.clone()
            } else {
                Self::compute_min_max(&all_features)?.0
            };
            let max = if let Some(ref max) = stats.max {
                max.clone()
            } else {
                Self::compute_min_max(&all_features)?.1
            };
            stats.histogram = Some(Self::compute_histogram(
                &all_features,
                &min,
                &max,
                config.histogram_bins,
            )?);
        }

        // Compute class distribution
        if config.compute_class_distribution {
            let mut class_counts = HashMap::new();
            for i in 0..sample_count {
                let (_, label) = dataset.get(i)?;
                let label_str = format!("{label:?}");
                *class_counts.entry(label_str).or_insert(0) += 1;
            }
            stats.class_distribution = Some(class_counts);
        }

        Ok(stats)
    }

    /// Convert tensor to vector
    pub fn tensor_to_vec<T>(tensor: &Tensor<T>) -> Result<Vec<T>>
    where
        T: Clone + Default + num_traits::Zero + Send + Sync + 'static,
    {
        // Get the raw data from tensor
        let data = tensor.as_slice().ok_or_else(|| {
            TensorError::invalid_argument(
                "Cannot access tensor data (GPU tensor not supported)".to_string(),
            )
        })?;
        Ok(data.to_vec())
    }

    /// Compute mean of feature vectors
    pub fn compute_mean<T>(features: &[Vec<T>]) -> Result<Vec<T>>
    where
        T: Clone + Default + num_traits::Zero + num_traits::Float,
    {
        if features.is_empty() {
            return Err(TensorError::invalid_argument(
                "Cannot compute mean of empty features".to_string(),
            ));
        }

        let feature_count = features[0].len();
        let mut mean = vec![T::zero(); feature_count];

        for feature_vec in features {
            for (i, &value) in feature_vec.iter().enumerate() {
                mean[i] = mean[i] + value;
            }
        }

        let n = T::from(features.len()).unwrap();
        for mean_val in &mut mean {
            *mean_val = *mean_val / n;
        }

        Ok(mean)
    }

    /// Compute standard deviation of feature vectors
    fn compute_std<T>(features: &[Vec<T>], mean: &[T]) -> Result<Vec<T>>
    where
        T: Clone + Default + num_traits::Zero + num_traits::Float,
    {
        if features.is_empty() {
            return Err(TensorError::invalid_argument(
                "Cannot compute std of empty features".to_string(),
            ));
        }

        let feature_count = features[0].len();
        let mut variance = vec![T::zero(); feature_count];

        for feature_vec in features {
            for (i, &value) in feature_vec.iter().enumerate() {
                let diff = value - mean[i];
                variance[i] = variance[i] + diff * diff;
            }
        }

        let n = T::from(features.len()).unwrap();
        let mut std = Vec::new();
        for var_val in variance {
            let std_val = (var_val / n).sqrt();
            std.push(std_val);
        }

        Ok(std)
    }

    /// Compute min and max of feature vectors
    fn compute_min_max<T>(features: &[Vec<T>]) -> Result<(Vec<T>, Vec<T>)>
    where
        T: Clone + Default + num_traits::Zero + num_traits::Float,
    {
        if features.is_empty() {
            return Err(TensorError::invalid_argument(
                "Cannot compute min/max of empty features".to_string(),
            ));
        }

        let _feature_count = features[0].len();
        let mut min_vals = features[0].clone();
        let mut max_vals = features[0].clone();

        for feature_vec in features.iter().skip(1) {
            for (i, &value) in feature_vec.iter().enumerate() {
                if value < min_vals[i] {
                    min_vals[i] = value;
                }
                if value > max_vals[i] {
                    max_vals[i] = value;
                }
            }
        }

        Ok((min_vals, max_vals))
    }

    /// Compute histogram of feature vectors
    fn compute_histogram<T>(
        features: &[Vec<T>],
        min_vals: &[T],
        max_vals: &[T],
        bins: usize,
    ) -> Result<Histogram<T>>
    where
        T: Clone + Default + num_traits::Zero + num_traits::Float,
    {
        if features.is_empty() {
            return Err(TensorError::invalid_argument(
                "Cannot compute histogram of empty features".to_string(),
            ));
        }

        // For simplicity, compute histogram for the first feature only
        let feature_idx = 0;
        let min_val = min_vals[feature_idx];
        let max_val = max_vals[feature_idx];

        // Create bin edges
        let mut bin_edges = Vec::new();
        let step = (max_val - min_val) / T::from(bins).unwrap();
        for i in 0..=bins {
            bin_edges.push(min_val + T::from(i).unwrap() * step);
        }

        // Count values in each bin
        let mut counts = vec![0usize; bins];
        for feature_vec in features {
            let value = feature_vec[feature_idx];
            let bin_idx = if value >= max_val {
                bins - 1
            } else {
                let normalized = (value - min_val) / (max_val - min_val);
                let idx = (normalized * T::from(bins).unwrap())
                    .to_usize()
                    .unwrap_or(0);
                idx.min(bins - 1)
            };
            counts[bin_idx] += 1;
        }

        // Create bin centers
        let mut bin_centers = Vec::new();
        for i in 0..bins {
            let center = (bin_edges[i] + bin_edges[i + 1]) / T::from(2).unwrap();
            bin_centers.push(center);
        }

        Ok(Histogram {
            bins: bin_centers,
            counts,
            bin_edges,
        })
    }
}
