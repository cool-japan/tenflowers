//! Correlation analysis for datasets
//!
//! This module provides functionality for computing correlation matrices
//! and analyzing relationships between dataset features.

use crate::Dataset;
use tenflowers_core::{Result, TensorError};

use super::computation::DatasetStatisticsComputer;

/// Correlation analysis for datasets
pub struct CorrelationAnalyzer;

impl CorrelationAnalyzer {
    /// Compute correlation matrix for dataset features
    pub fn compute_correlation_matrix<T, D>(dataset: &D) -> Result<Vec<Vec<T>>>
    where
        T: Clone
            + Default
            + scirs2_core::numeric::Zero
            + scirs2_core::numeric::Float
            + std::fmt::Debug
            + Send
            + Sync
            + 'static,
        D: Dataset<T>,
    {
        if dataset.is_empty() {
            return Err(TensorError::invalid_argument(
                "Cannot compute correlation on empty dataset".to_string(),
            ));
        }

        let sample_count = dataset.len();
        let first_sample = dataset.get(0)?;
        let feature_count = first_sample.0.shape().dims().iter().product::<usize>();

        // Collect all feature vectors
        let mut all_features = Vec::new();
        for i in 0..sample_count {
            let (features, _) = dataset.get(i)?;
            let feature_vec = DatasetStatisticsComputer::tensor_to_vec(&features)?;
            all_features.push(feature_vec);
        }

        // Compute means
        let means = DatasetStatisticsComputer::compute_mean(&all_features)?;

        // Compute correlation matrix
        let mut correlation_matrix = vec![vec![T::zero(); feature_count]; feature_count];

        for (i, row) in correlation_matrix.iter_mut().enumerate() {
            for j in 0..feature_count {
                let corr = Self::compute_correlation(&all_features, i, j, &means)?;
                row[j] = corr;
            }
        }

        Ok(correlation_matrix)
    }

    /// Compute correlation between two features
    fn compute_correlation<T>(
        features: &[Vec<T>],
        feature_i: usize,
        feature_j: usize,
        means: &[T],
    ) -> Result<T>
    where
        T: Clone + Default + scirs2_core::numeric::Zero + scirs2_core::numeric::Float,
    {
        let _n = T::from(features.len()).unwrap();
        let mean_i = means[feature_i];
        let mean_j = means[feature_j];

        let mut numerator = T::zero();
        let mut sum_sq_i = T::zero();
        let mut sum_sq_j = T::zero();

        for feature_vec in features {
            let diff_i = feature_vec[feature_i] - mean_i;
            let diff_j = feature_vec[feature_j] - mean_j;

            numerator = numerator + diff_i * diff_j;
            sum_sq_i = sum_sq_i + diff_i * diff_i;
            sum_sq_j = sum_sq_j + diff_j * diff_j;
        }

        let denominator = (sum_sq_i * sum_sq_j).sqrt();

        if denominator.is_zero() {
            Ok(T::zero())
        } else {
            Ok(numerator / denominator)
        }
    }
}
