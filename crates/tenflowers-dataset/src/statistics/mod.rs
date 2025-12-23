//! Dataset statistics computation module
//!
//! This module provides utilities for computing various statistics on datasets
//! including mean, standard deviation, histograms, and class distributions.
//!
//! The module is organized into several submodules:
//! - `core`: Core types and data structures
//! - `computation`: Statistics computation algorithms
//! - `correlation`: Correlation analysis functionality
//! - `advanced`: Advanced statistical methods using SciRS2
//! - `extensions`: Convenient trait extensions for datasets

#![allow(clippy::needless_range_loop)]

pub mod advanced;
pub mod computation;
pub mod core;
pub mod correlation;
pub mod extensions;

// Re-export all public types and traits for convenience
pub use advanced::{AdvancedStatistics, AdvancedStatisticsExt, MultivariateStatistics, PCAResult};
pub use computation::DatasetStatisticsComputer;
pub use core::{DatasetStats, Histogram, StatisticsConfig};
pub use correlation::CorrelationAnalyzer;
pub use extensions::DatasetStatisticsExt;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TensorDataset;
    use tenflowers_core::Tensor;

    #[test]
    fn test_statistics_config_default() {
        let config = StatisticsConfig::default();
        assert!(config.compute_mean);
        assert!(config.compute_std);
        assert!(config.compute_min_max);
        assert!(!config.compute_histogram);
        assert!(!config.compute_class_distribution);
        assert_eq!(config.histogram_bins, 50);
    }

    #[test]
    fn test_dataset_statistics_computation() {
        // Create test dataset
        let features =
            Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();
        let labels = Tensor::<f32>::from_vec(vec![0.0, 1.0, 2.0], &[3]).unwrap();

        let dataset = TensorDataset::new(features, labels);

        // Compute statistics
        let stats = dataset.compute_statistics().unwrap();

        assert_eq!(stats.sample_count(), 3);
        assert_eq!(stats.feature_count(), 2);
        assert!(stats.has_mean());
        assert!(stats.has_std());
        assert!(stats.has_min_max());

        // Check mean values
        let mean = stats.mean.unwrap();
        assert_eq!(mean.len(), 2);
        assert!((mean[0] - 3.0).abs() < 1e-6); // (1+3+5)/3 = 3
        assert!((mean[1] - 4.0).abs() < 1e-6); // (2+4+6)/3 = 4
    }

    #[test]
    fn test_dataset_statistics_with_config() {
        // Create test dataset
        let features = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let labels = Tensor::<f32>::from_vec(vec![0.0, 1.0], &[2]).unwrap();

        let dataset = TensorDataset::new(features, labels);

        // Compute statistics with custom config
        let config = StatisticsConfig {
            compute_mean: true,
            compute_std: false,
            compute_min_max: true,
            compute_histogram: true,
            compute_class_distribution: true,
            histogram_bins: 10,
        };

        let stats = dataset.compute_statistics_with_config(config).unwrap();

        assert!(stats.has_mean());
        assert!(!stats.has_std());
        assert!(stats.has_min_max());
        assert!(stats.has_histogram());
        assert!(stats.has_class_distribution());

        // Check class distribution
        let class_dist = stats.class_distribution.unwrap();
        assert_eq!(class_dist.len(), 2);
    }

    #[test]
    fn test_correlation_analysis() {
        // Create test dataset with correlated features
        let features =
            Tensor::<f32>::from_vec(vec![1.0, 2.0, 2.0, 4.0, 3.0, 6.0], &[3, 2]).unwrap();
        let labels = Tensor::<f32>::from_vec(vec![0.0, 1.0, 2.0], &[3]).unwrap();

        let dataset = TensorDataset::new(features, labels);

        // Compute correlation matrix
        let correlation_matrix = CorrelationAnalyzer::compute_correlation_matrix(&dataset).unwrap();

        assert_eq!(correlation_matrix.len(), 2);
        assert_eq!(correlation_matrix[0].len(), 2);

        // Diagonal should be 1.0 (perfect correlation with itself)
        assert!((correlation_matrix[0][0] - 1.0).abs() < 1e-6);
        assert!((correlation_matrix[1][1] - 1.0).abs() < 1e-6);

        // Off-diagonal should be the correlation between features
        assert!((correlation_matrix[0][1] - correlation_matrix[1][0]).abs() < 1e-6);
    }
}

#[cfg(test)]
pub mod advanced_statistics_tests {
    use super::*;
    use crate::Dataset;
    use scirs2_core::random::rng;
    use tenflowers_core::{Result, Tensor, TensorError};

    /// Simple dataset implementation for testing
    pub struct TensorDataset<T> {
        features: Tensor<T>,
        labels: Tensor<T>,
    }

    impl<T> TensorDataset<T> {
        pub fn new(features: Tensor<T>, labels: Tensor<T>) -> Self {
            Self { features, labels }
        }
    }

    impl<
            T: Clone
                + Default
                + scirs2_core::numeric::Float
                + Send
                + Sync
                + bytemuck::Pod
                + bytemuck::Zeroable
                + 'static,
        > Dataset<T> for TensorDataset<T>
    {
        fn len(&self) -> usize {
            self.features.shape().dims()[0]
        }

        fn get(&self, index: usize) -> Result<(Tensor<T>, Tensor<T>)> {
            if index >= self.len() {
                return Err(TensorError::invalid_argument(format!(
                    "Index {} out of bounds",
                    index
                )));
            }

            // Extract the specific sample by slicing
            let feature_shape = self.features.shape();
            let label_shape = self.labels.shape();

            // For features: get the slice for this specific sample
            // If features are [n_samples, n_features], we want features[index, :]
            let feature_sample = if feature_shape.rank() == 2 {
                // Create a view of the single sample (features[index, :])
                let sample_size = feature_shape.dims()[1];
                let start_idx = index * sample_size;
                let end_idx = start_idx + sample_size;
                let feature_data = self.features.to_vec()?;
                Tensor::from_vec(feature_data[start_idx..end_idx].to_vec(), &[sample_size])?
            } else {
                // For 1D case, just get the single element
                let feature_data = self.features.to_vec()?;
                Tensor::from_vec(vec![feature_data[index]], &[1])?
            };

            // For labels: get the single label value
            let label_sample = if label_shape.rank() == 1 {
                let label_data = self.labels.to_vec()?;
                Tensor::from_vec(vec![label_data[index]], &[1])?
            } else {
                // For higher dimensions, similar slicing would be needed
                self.labels.clone()
            };

            Ok((feature_sample, label_sample))
        }
    }

    #[test]
    fn test_multivariate_statistics_computation() {
        // Create test dataset with correlated features
        let n_samples = 20;
        let n_features = 3;
        let feature_data = vec![
            // Sample features with some correlation
            1.0, 2.0, 1.5, 2.0, 4.0, 3.0, 3.0, 6.0, 4.5, 4.0, 8.0, 6.0, 5.0, 10.0, 7.5, 1.5, 3.0,
            2.25, 2.5, 5.0, 3.75, 3.5, 7.0, 5.25, 4.5, 9.0, 6.75, 5.5, 11.0, 8.25, 1.2, 2.4, 1.8,
            2.2, 4.4, 3.3, 3.2, 6.4, 4.8, 4.2, 8.4, 6.3, 5.2, 10.4, 7.8, 1.8, 3.6, 2.7, 2.8, 5.6,
            4.2, 3.8, 7.6, 5.7, 4.8, 9.6, 7.2, 5.8, 11.6, 8.7,
        ];

        let features = Tensor::<f32>::from_vec(feature_data, &[n_samples, n_features]).unwrap();
        let labels = Tensor::<f32>::zeros(&[n_samples]);

        let dataset = TensorDataset::new(features, labels);

        // Test multivariate statistics
        let multivariate_stats = dataset.compute_multivariate_statistics().unwrap();
        assert_eq!(multivariate_stats.n_features, n_features);
        assert_eq!(multivariate_stats.n_samples, n_samples);
        assert_eq!(multivariate_stats.covariance_matrix.len(), n_features);
        assert_eq!(multivariate_stats.eigenvalues.len(), n_features);

        println!("Multivariate statistics computation completed successfully!");
        println!(
            "Covariance matrix dimensions: {}x{}",
            multivariate_stats.covariance_matrix.len(),
            multivariate_stats.covariance_matrix[0].len()
        );
        println!(
            "Number of eigenvalues: {}",
            multivariate_stats.eigenvalues.len()
        );
    }

    #[test]
    fn test_pca_computation() {
        // Create test dataset with correlated features
        let n_samples = 30;
        let feature_data = vec![
            // Sample 1: [1.0, 2.0], Sample 2: [2.0, 4.0], etc.
            1.0, 2.0, 2.0, 4.0, 3.0, 6.0, 4.0, 8.0, 5.0, 10.0, 1.5, 3.0, 2.5, 5.0, 3.5, 7.0, 4.5,
            9.0, 5.5, 11.0, 2.0, 4.0, 3.0, 6.0, 4.0, 8.0, 5.0, 10.0, 6.0, 12.0, 1.2, 2.4, 2.2, 4.4,
            3.2, 6.4, 4.2, 8.4, 5.2, 10.4, 1.8, 3.6, 2.8, 5.6, 3.8, 7.6, 4.8, 9.6, 5.8, 11.6, 2.1,
            4.2, 3.1, 6.2, 4.1, 8.2, 5.1, 10.2, 6.1, 12.2,
        ];

        let features = Tensor::<f32>::from_vec(feature_data, &[n_samples, 2]).unwrap();
        let labels = Tensor::<f32>::zeros(&[n_samples]);

        let dataset = TensorDataset::new(features, labels);

        let pca_result = dataset.compute_pca(2).unwrap();

        // Verify PCA results
        assert_eq!(pca_result.n_components, 2);
        assert_eq!(pca_result.n_features, 2);
        assert_eq!(pca_result.principal_components.len(), 2);
        assert_eq!(pca_result.explained_variance_ratio.len(), 2);

        // Variance ratios should sum to <= 1.0
        let total_variance: f32 = pca_result.explained_variance_ratio.iter().sum();
        assert!(total_variance <= 1.0001); // Allow for floating point precision

        println!("PCA computed successfully!");
        println!(
            "Principal components: {:?}",
            pca_result.principal_components
        );
        println!(
            "Explained variance ratio: {:?}",
            pca_result.explained_variance_ratio
        );
    }

    #[test]
    fn test_advanced_statistics_with_random_data() {
        // Generate random dataset for robustness testing
        use scirs2_core::random::rand_prelude::*;
        let mut rng = rng();
        let n_samples = 100;
        let n_features = 4;

        let mut feature_data = Vec::with_capacity(n_samples * n_features);
        for _ in 0..n_samples * n_features {
            let random_val: f32 = rng.random();
            feature_data.push(random_val * 10.0 - 5.0); // Scale to [-5.0, 5.0]
        }

        let features = Tensor::<f32>::from_vec(feature_data, &[n_samples, n_features]).unwrap();
        let labels = Tensor::<f32>::zeros(&[n_samples]);

        let dataset = TensorDataset::new(features, labels);

        // Test multivariate statistics
        let multivariate_stats = dataset.compute_multivariate_statistics().unwrap();
        assert_eq!(multivariate_stats.n_features, n_features);
        assert_eq!(multivariate_stats.n_samples, n_samples);

        // Test PCA with fewer components than features
        let pca_result = dataset.compute_pca(2).unwrap();
        assert_eq!(pca_result.n_components, 2);
        assert!(pca_result.explained_variance_ratio.len() <= n_features);

        println!("Advanced statistics with random data completed successfully!");
    }
}
