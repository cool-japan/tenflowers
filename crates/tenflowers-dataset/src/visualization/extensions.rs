//! Extension traits for easy visualization access
//!
//! This module provides convenient trait extensions that add visualization
//! methods directly to datasets.

use crate::{transforms::Transform, Dataset};
use tenflowers_core::Result;

use super::types::*;
use super::visualizer::DatasetVisualizer;

/// Extension trait for easy visualization access
pub trait DatasetVisualizationExt<T>: Dataset<T> + Sized {
    /// Create a sample preview
    fn sample_preview(&self, num_samples: usize) -> Result<SamplePreview>
    where
        T: Clone + Default + scirs2_core::numeric::Zero + Send + Sync + 'static,
    {
        DatasetVisualizer::sample_preview(self, num_samples)
    }

    /// Get feature distribution information
    fn feature_distribution(&self, max_samples: Option<usize>) -> Result<DistributionInfo<T>>
    where
        T: Clone
            + Default
            + scirs2_core::numeric::Zero
            + Send
            + Sync
            + 'static
            + scirs2_core::numeric::Float,
    {
        DatasetVisualizer::feature_distribution(self, max_samples)
    }

    /// Get class distribution
    fn class_distribution(&self) -> Result<ClassDistribution>
    where
        T: Clone + Default + scirs2_core::numeric::Zero + Send + Sync + 'static,
    {
        DatasetVisualizer::class_distribution(self)
    }

    /// Create a histogram for a specific feature
    fn feature_histogram(&self, feature_index: usize, bins: usize) -> Result<FeatureHistogram<T>>
    where
        T: Clone
            + Default
            + scirs2_core::numeric::Zero
            + Send
            + Sync
            + 'static
            + scirs2_core::numeric::Float
            + PartialOrd,
    {
        DatasetVisualizer::feature_histogram(self, feature_index, bins)
    }

    /// Analyze the effects of a transform on dataset samples
    fn analyze_augmentation_effects<Tr>(
        &self,
        transform: &Tr,
        num_samples: usize,
    ) -> Result<AugmentationEffects<T>>
    where
        T: Clone
            + Default
            + scirs2_core::numeric::Zero
            + Send
            + Sync
            + 'static
            + scirs2_core::numeric::Float
            + PartialOrd,
        Tr: Transform<T>,
    {
        DatasetVisualizer::analyze_augmentation_effects(self, transform, num_samples)
    }
}

// Implement the extension trait for all datasets
impl<T, D: Dataset<T> + Sized> DatasetVisualizationExt<T> for D {}
