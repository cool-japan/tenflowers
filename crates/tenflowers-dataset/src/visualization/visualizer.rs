//! Core dataset visualization implementation
//!
//! This module contains the main DatasetVisualizer struct and its methods
//! for analyzing and visualizing dataset properties.

use crate::{transforms::Transform, Dataset};
use tenflowers_core::{Result, TensorError};

use super::types::*;

/// Visualization utilities for datasets
pub struct DatasetVisualizer;

impl DatasetVisualizer {
    /// Create a sample preview showing basic statistics and examples
    pub fn sample_preview<T, D>(dataset: &D, num_samples: usize) -> Result<SamplePreview>
    where
        T: Clone + Default + num_traits::Zero + Send + Sync + 'static,
        D: Dataset<T>,
    {
        if dataset.is_empty() {
            return Err(TensorError::invalid_argument(
                "Dataset is empty".to_string(),
            ));
        }

        let total_samples = dataset.len();
        let samples_to_show = num_samples.min(total_samples);

        // Get sample data
        let mut samples = Vec::new();
        let step = if samples_to_show == 1 {
            0
        } else {
            total_samples / samples_to_show
        };

        for i in 0..samples_to_show {
            let index = if step == 0 { 0 } else { i * step };
            let index = index.min(total_samples - 1);

            if let Ok((features, labels)) = dataset.get(index) {
                samples.push(SampleInfo {
                    index,
                    feature_shape: features.shape().dims().to_vec(),
                    label_shape: labels.shape().dims().to_vec(),
                });
            }
        }

        Ok(SamplePreview {
            total_samples,
            samples_shown: samples.len(),
            samples,
        })
    }

    /// Generate distribution information for dataset features and labels
    pub fn feature_distribution<T, D>(
        dataset: &D,
        max_samples: Option<usize>,
    ) -> Result<DistributionInfo<T>>
    where
        T: Clone + Default + num_traits::Zero + Send + Sync + 'static + num_traits::Float,
        D: Dataset<T>,
    {
        if dataset.is_empty() {
            return Err(TensorError::invalid_argument(
                "Dataset is empty".to_string(),
            ));
        }

        let samples_to_analyze = max_samples.unwrap_or(dataset.len()).min(dataset.len());
        let mut feature_stats = Vec::new();
        let mut label_stats = Vec::new();

        // Get first sample to determine shapes
        let (first_features, first_labels) = dataset.get(0)?;
        let feature_dims = first_features.numel();
        let label_dims = first_labels.numel();

        // Initialize accumulators
        let mut feature_sums = vec![T::zero(); feature_dims];
        let mut feature_squared_sums = vec![T::zero(); feature_dims];
        let mut label_sums = vec![T::zero(); label_dims];
        let mut label_squared_sums = vec![T::zero(); label_dims];

        let mut valid_samples = 0;

        // Accumulate statistics
        for i in 0..samples_to_analyze {
            if let Ok((features, labels)) = dataset.get(i) {
                // Process features
                if let Some(feature_data) = features.as_slice() {
                    for (j, &value) in feature_data.iter().enumerate() {
                        if j < feature_dims {
                            feature_sums[j] = feature_sums[j] + value;
                            feature_squared_sums[j] = feature_squared_sums[j] + value * value;
                        }
                    }
                }

                // Process labels
                if let Some(label_data) = labels.as_slice() {
                    for (j, &value) in label_data.iter().enumerate() {
                        if j < label_dims {
                            label_sums[j] = label_sums[j] + value;
                            label_squared_sums[j] = label_squared_sums[j] + value * value;
                        }
                    }
                }

                valid_samples += 1;
            }
        }

        if valid_samples == 0 {
            return Err(TensorError::invalid_argument(
                "No valid samples found".to_string(),
            ));
        }

        let n = T::from(valid_samples).unwrap();

        // Calculate feature statistics
        for i in 0..feature_dims {
            let mean = feature_sums[i] / n;
            let variance = (feature_squared_sums[i] / n) - (mean * mean);
            let std_dev = variance.sqrt();

            feature_stats.push(FeatureStats {
                dimension: i,
                mean,
                std_dev,
                min: T::zero(), // Would need to track min/max separately
                max: T::zero(),
            });
        }

        // Calculate label statistics
        for i in 0..label_dims {
            let mean = label_sums[i] / n;
            let variance = (label_squared_sums[i] / n) - (mean * mean);
            let std_dev = variance.sqrt();

            label_stats.push(FeatureStats {
                dimension: i,
                mean,
                std_dev,
                min: T::zero(),
                max: T::zero(),
            });
        }

        Ok(DistributionInfo {
            samples_analyzed: valid_samples,
            feature_stats,
            label_stats,
        })
    }

    /// Generate class distribution for classification datasets
    pub fn class_distribution<T, D>(dataset: &D) -> Result<ClassDistribution>
    where
        T: Clone + Default + num_traits::Zero + Send + Sync + 'static,
        D: Dataset<T>,
    {
        let mut class_counts = std::collections::HashMap::new();
        let mut total_samples = 0;

        for i in 0..dataset.len() {
            if let Ok((_, labels)) = dataset.get(i) {
                // For simplicity, convert labels to string representation
                let class_key = format!("{:?}", labels.shape());
                *class_counts.entry(class_key).or_insert(0) += 1;
                total_samples += 1;
            }
        }

        Ok(ClassDistribution {
            total_samples,
            class_counts,
        })
    }

    /// Generate a simple text-based histogram for a single feature dimension
    pub fn feature_histogram<T, D>(
        dataset: &D,
        feature_index: usize,
        bins: usize,
    ) -> Result<FeatureHistogram<T>>
    where
        T: Clone
            + Default
            + num_traits::Zero
            + Send
            + Sync
            + 'static
            + num_traits::Float
            + PartialOrd,
        D: Dataset<T>,
    {
        let mut values = Vec::new();

        // Collect all values for the specified feature
        for i in 0..dataset.len() {
            if let Ok((features, _)) = dataset.get(i) {
                if let Some(feature_data) = features.as_slice() {
                    if feature_index < feature_data.len() {
                        values.push(feature_data[feature_index]);
                    }
                }
            }
        }

        if values.is_empty() {
            return Err(TensorError::invalid_argument(
                "No valid feature values found".to_string(),
            ));
        }

        // Find min and max
        let mut min_val = values[0];
        let mut max_val = values[0];

        for &val in &values {
            if val < min_val {
                min_val = val;
            }
            if val > max_val {
                max_val = val;
            }
        }

        // Create bins
        let range = max_val - min_val;
        let bin_width = if range > T::zero() {
            range / T::from(bins).unwrap()
        } else {
            T::from(1.0).unwrap()
        };

        let mut bin_counts = vec![0; bins];

        // Assign values to bins
        for val in values {
            if range > T::zero() {
                let bin_index = ((val - min_val) / bin_width).to_usize().unwrap_or(0);
                let bin_index = bin_index.min(bins - 1);
                bin_counts[bin_index] += 1;
            } else {
                bin_counts[0] += 1;
            }
        }

        Ok(FeatureHistogram {
            feature_index,
            min_value: min_val,
            max_value: max_val,
            bin_width,
            bin_counts,
        })
    }

    /// Analyze the effects of a transform on dataset samples
    pub fn analyze_augmentation_effects<T, D, Tr>(
        dataset: &D,
        transform: &Tr,
        num_samples: usize,
    ) -> Result<AugmentationEffects<T>>
    where
        T: Clone
            + Default
            + num_traits::Zero
            + Send
            + Sync
            + 'static
            + num_traits::Float
            + PartialOrd,
        D: Dataset<T>,
        Tr: Transform<T>,
    {
        if dataset.is_empty() {
            return Err(TensorError::invalid_argument(
                "Dataset is empty".to_string(),
            ));
        }

        let samples_to_analyze = num_samples.min(dataset.len());
        let mut before_after_pairs = Vec::new();
        let mut transform_success_count = 0;

        // Collect before/after pairs
        for i in 0..samples_to_analyze {
            if let Ok(original_sample) = dataset.get(i) {
                match transform.apply(original_sample.clone()) {
                    Ok(transformed_sample) => {
                        before_after_pairs.push(BeforeAfterPair {
                            index: i,
                            original: original_sample,
                            transformed: transformed_sample,
                        });
                        transform_success_count += 1;
                    }
                    Err(_) => {
                        // Transform failed, skip this sample
                        continue;
                    }
                }
            }
        }

        if before_after_pairs.is_empty() {
            return Err(TensorError::invalid_argument(
                "No successful transforms".to_string(),
            ));
        }

        // Analyze feature changes
        let feature_changes = Self::analyze_feature_changes(&before_after_pairs)?;

        // Analyze distribution changes
        let distribution_changes = Self::analyze_distribution_changes(&before_after_pairs)?;

        Ok(AugmentationEffects {
            samples_analyzed: before_after_pairs.len(),
            transform_success_rate: transform_success_count as f64 / samples_to_analyze as f64,
            feature_changes,
            distribution_changes,
            sample_pairs: before_after_pairs,
        })
    }

    /// Compare before/after samples for a specific transform
    pub fn compare_samples<T, Tr>(
        samples: &[(tenflowers_core::Tensor<T>, tenflowers_core::Tensor<T>)],
        transform: &Tr,
        comparison_count: usize,
    ) -> Result<Vec<SampleComparison<T>>>
    where
        T: Clone + Default + num_traits::Zero + Send + Sync + 'static + num_traits::Float,
        Tr: Transform<T>,
    {
        let mut comparisons = Vec::new();
        let samples_to_compare = comparison_count.min(samples.len());

        for (i, original) in samples.iter().enumerate().take(samples_to_compare) {
            let original = original.clone();

            match transform.apply(original.clone()) {
                Ok(transformed) => {
                    // Calculate basic statistics
                    let original_stats = Self::calculate_tensor_stats(&original.0)?;
                    let transformed_stats = Self::calculate_tensor_stats(&transformed.0)?;

                    comparisons.push(SampleComparison {
                        sample_index: i,
                        original_stats,
                        transformed_stats,
                        change_magnitude: Self::calculate_change_magnitude(
                            &original.0,
                            &transformed.0,
                        )?,
                    });
                }
                Err(_) => {
                    // Skip failed transforms
                    continue;
                }
            }
        }

        Ok(comparisons)
    }

    // Helper method to analyze feature changes across all samples
    pub fn analyze_feature_changes<T>(
        pairs: &[BeforeAfterPair<T>],
    ) -> Result<FeatureChangeAnalysis<T>>
    where
        T: Clone + Default + num_traits::Zero + Send + Sync + 'static + num_traits::Float,
    {
        if pairs.is_empty() {
            return Err(TensorError::invalid_argument(
                "No sample pairs provided".to_string(),
            ));
        }

        // Get feature dimensions from first sample
        let first_features = &pairs[0].original.0;
        let feature_count = first_features.numel();

        let mut total_change = T::zero();
        let mut max_change = T::zero();
        let mut min_change = T::from(f64::INFINITY).unwrap_or(T::zero());
        let mut change_count = 0;

        // Calculate changes across all samples
        for pair in pairs {
            if let (Some(orig_data), Some(trans_data)) =
                (pair.original.0.as_slice(), pair.transformed.0.as_slice())
            {
                for (orig, trans) in orig_data.iter().zip(trans_data.iter()) {
                    let change = (*trans - *orig).abs();
                    total_change = total_change + change;

                    if change > max_change {
                        max_change = change;
                    }
                    if change < min_change {
                        min_change = change;
                    }
                    change_count += 1;
                }
            }
        }

        let avg_change = if change_count > 0 {
            total_change / T::from(change_count).unwrap_or(T::from(1.0).unwrap())
        } else {
            T::zero()
        };

        Ok(FeatureChangeAnalysis {
            feature_count,
            average_change: avg_change,
            max_change,
            min_change,
            samples_with_changes: pairs.len(),
        })
    }

    // Helper method to analyze distribution changes
    pub fn analyze_distribution_changes<T>(
        pairs: &[BeforeAfterPair<T>],
    ) -> Result<DistributionChangeAnalysis<T>>
    where
        T: Clone + Default + num_traits::Zero + Send + Sync + 'static + num_traits::Float,
    {
        // Calculate mean and std before and after transformation
        let mut original_sum = T::zero();
        let mut transformed_sum = T::zero();
        let mut original_squared_sum = T::zero();
        let mut transformed_squared_sum = T::zero();
        let mut total_elements = 0;

        for pair in pairs {
            if let (Some(orig_data), Some(trans_data)) =
                (pair.original.0.as_slice(), pair.transformed.0.as_slice())
            {
                for (&orig, &trans) in orig_data.iter().zip(trans_data.iter()) {
                    original_sum = original_sum + orig;
                    transformed_sum = transformed_sum + trans;
                    original_squared_sum = original_squared_sum + orig * orig;
                    transformed_squared_sum = transformed_squared_sum + trans * trans;
                    total_elements += 1;
                }
            }
        }

        if total_elements == 0 {
            return Err(TensorError::invalid_argument(
                "No valid data found".to_string(),
            ));
        }

        let n = T::from(total_elements).unwrap_or(T::from(1.0).unwrap());

        let original_mean = original_sum / n;
        let transformed_mean = transformed_sum / n;

        let original_variance = (original_squared_sum / n) - (original_mean * original_mean);
        let transformed_variance =
            (transformed_squared_sum / n) - (transformed_mean * transformed_mean);

        let original_std = original_variance.sqrt();
        let transformed_std = transformed_variance.sqrt();

        Ok(DistributionChangeAnalysis {
            original_mean,
            transformed_mean,
            original_std,
            transformed_std,
            mean_change: (transformed_mean - original_mean).abs(),
            std_change: (transformed_std - original_std).abs(),
        })
    }

    // Helper method to calculate basic tensor statistics
    pub fn calculate_tensor_stats<T>(tensor: &tenflowers_core::Tensor<T>) -> Result<TensorStats<T>>
    where
        T: Clone + Default + num_traits::Zero + Send + Sync + 'static + num_traits::Float,
    {
        if let Some(data) = tensor.as_slice() {
            if data.is_empty() {
                return Ok(TensorStats {
                    mean: T::zero(),
                    std: T::zero(),
                    min: T::zero(),
                    max: T::zero(),
                    element_count: 0,
                });
            }

            let mut sum = T::zero();
            let mut squared_sum = T::zero();
            let mut min_val = data[0];
            let mut max_val = data[0];

            for &value in data {
                sum = sum + value;
                squared_sum = squared_sum + value * value;
                if value < min_val {
                    min_val = value;
                }
                if value > max_val {
                    max_val = value;
                }
            }

            let n = T::from(data.len()).unwrap_or(T::from(1.0).unwrap());
            let mean = sum / n;
            let variance = (squared_sum / n) - (mean * mean);
            let std = variance.sqrt();

            Ok(TensorStats {
                mean,
                std,
                min: min_val,
                max: max_val,
                element_count: data.len(),
            })
        } else {
            Err(TensorError::device_error_simple(
                "Cannot access tensor data".to_string(),
            ))
        }
    }

    // Helper method to calculate change magnitude between tensors
    pub fn calculate_change_magnitude<T>(
        original: &tenflowers_core::Tensor<T>,
        transformed: &tenflowers_core::Tensor<T>,
    ) -> Result<T>
    where
        T: Clone + Default + num_traits::Zero + Send + Sync + 'static + num_traits::Float,
    {
        if let (Some(orig_data), Some(trans_data)) = (original.as_slice(), transformed.as_slice()) {
            if orig_data.len() != trans_data.len() {
                return Err(TensorError::invalid_argument(
                    "Tensor size mismatch".to_string(),
                ));
            }

            let mut total_change = T::zero();
            for (orig, trans) in orig_data.iter().zip(trans_data.iter()) {
                let diff = *trans - *orig;
                total_change = total_change + diff * diff;
            }

            let n = T::from(orig_data.len()).unwrap_or(T::from(1.0).unwrap());
            Ok((total_change / n).sqrt()) // RMS change
        } else {
            Err(TensorError::device_error_simple(
                "Cannot access tensor data".to_string(),
            ))
        }
    }
}
