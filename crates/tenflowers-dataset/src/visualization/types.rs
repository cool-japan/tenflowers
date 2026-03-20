//! Data structures for dataset visualization
//!
//! This module contains all the data types used for representing
//! visualization information about datasets.

use std::collections::HashMap;

/// Sample preview information
#[derive(Debug, Clone)]
pub struct SamplePreview {
    pub total_samples: usize,
    pub samples_shown: usize,
    pub samples: Vec<SampleInfo>,
}

/// Information about a single sample
#[derive(Debug, Clone)]
pub struct SampleInfo {
    pub index: usize,
    pub feature_shape: Vec<usize>,
    pub label_shape: Vec<usize>,
}

/// Distribution information for features and labels
#[derive(Debug, Clone)]
pub struct DistributionInfo<T> {
    pub samples_analyzed: usize,
    pub feature_stats: Vec<FeatureStats<T>>,
    pub label_stats: Vec<FeatureStats<T>>,
}

/// Statistics for a single feature dimension
#[derive(Debug, Clone)]
pub struct FeatureStats<T> {
    pub dimension: usize,
    pub mean: T,
    pub std_dev: T,
    pub min: T,
    pub max: T,
}

/// Class distribution information
#[derive(Debug, Clone)]
pub struct ClassDistribution {
    pub total_samples: usize,
    pub class_counts: HashMap<String, usize>,
}

/// Histogram information for a feature
#[derive(Debug, Clone)]
pub struct FeatureHistogram<T> {
    pub feature_index: usize,
    pub min_value: T,
    pub max_value: T,
    pub bin_width: T,
    pub bin_counts: Vec<usize>,
}

/// Analysis of augmentation effects on dataset samples
#[derive(Debug, Clone)]
pub struct AugmentationEffects<T> {
    pub samples_analyzed: usize,
    pub transform_success_rate: f64,
    pub feature_changes: FeatureChangeAnalysis<T>,
    pub distribution_changes: DistributionChangeAnalysis<T>,
    pub sample_pairs: Vec<BeforeAfterPair<T>>,
}

/// Before/after pair for transformation analysis
#[derive(Debug, Clone)]
pub struct BeforeAfterPair<T> {
    pub index: usize,
    pub original: (tenflowers_core::Tensor<T>, tenflowers_core::Tensor<T>),
    pub transformed: (tenflowers_core::Tensor<T>, tenflowers_core::Tensor<T>),
}

/// Analysis of feature changes from transformations
#[derive(Debug, Clone)]
pub struct FeatureChangeAnalysis<T> {
    pub feature_count: usize,
    pub average_change: T,
    pub max_change: T,
    pub min_change: T,
    pub samples_with_changes: usize,
}

/// Analysis of distribution changes from transformations
#[derive(Debug, Clone)]
pub struct DistributionChangeAnalysis<T> {
    pub original_mean: T,
    pub transformed_mean: T,
    pub original_std: T,
    pub transformed_std: T,
    pub mean_change: T,
    pub std_change: T,
}

/// Comparison of individual samples before/after transformation
#[derive(Debug, Clone)]
pub struct SampleComparison<T> {
    pub sample_index: usize,
    pub original_stats: TensorStats<T>,
    pub transformed_stats: TensorStats<T>,
    pub change_magnitude: T,
}

/// Basic statistics for a tensor
#[derive(Debug, Clone)]
pub struct TensorStats<T> {
    pub mean: T,
    pub std: T,
    pub min: T,
    pub max: T,
    pub element_count: usize,
}
