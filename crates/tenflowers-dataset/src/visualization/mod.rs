//! Dataset visualization module
//!
//! This module provides utilities for visualizing and analyzing datasets
//! including sample previews, distribution analysis, histograms, and
//! augmentation effect analysis.
//!
//! The module is organized into several submodules:
//! - `types`: Core data structures for visualization information
//! - `visualizer`: Main DatasetVisualizer implementation with analysis methods
//! - `formatting`: Display and formatting implementations for all types
//! - `extensions`: Convenient trait extensions for datasets

pub mod extensions;
pub mod formatting;
pub mod types;
pub mod visualizer;

// Re-export all public types and traits for convenience
pub use extensions::DatasetVisualizationExt;
pub use types::*;
pub use visualizer::DatasetVisualizer;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TensorDataset;
    use tenflowers_core::Tensor;

    #[test]
    fn test_sample_preview() {
        let features =
            Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();
        let labels = Tensor::<f32>::from_vec(vec![0.0, 1.0, 2.0], &[3]).unwrap();

        let dataset = TensorDataset::new(features, labels);
        let preview = dataset.sample_preview(2).unwrap();

        assert_eq!(preview.total_samples, 3);
        assert!(preview.samples_shown <= 2);
        assert!(!preview.samples.is_empty());
    }

    #[test]
    fn test_feature_distribution() {
        let features =
            Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();
        let labels = Tensor::<f32>::from_vec(vec![0.0, 1.0, 2.0], &[3]).unwrap();

        let dataset = TensorDataset::new(features, labels);
        let distribution = dataset.feature_distribution(None).unwrap();

        assert_eq!(distribution.samples_analyzed, 3);
        assert_eq!(distribution.feature_stats.len(), 2); // 2 feature dimensions
        assert_eq!(distribution.label_stats.len(), 1); // 1 label dimension
    }

    #[test]
    fn test_class_distribution() {
        let features = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let labels = Tensor::<f32>::from_vec(vec![0.0, 1.0], &[2]).unwrap();

        let dataset = TensorDataset::new(features, labels);
        let class_dist = dataset.class_distribution().unwrap();

        assert_eq!(class_dist.total_samples, 2);
        assert!(!class_dist.class_counts.is_empty());
    }

    #[test]
    fn test_feature_histogram() {
        let features =
            Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[4, 2]).unwrap();
        let labels = Tensor::<f32>::from_vec(vec![0.0, 1.0, 2.0, 3.0], &[4]).unwrap();

        let dataset = TensorDataset::new(features, labels);
        let histogram = dataset.feature_histogram(0, 3).unwrap();

        assert_eq!(histogram.feature_index, 0);
        assert_eq!(histogram.bin_counts.len(), 3);
        assert!(histogram.max_value >= histogram.min_value);
    }

    #[test]
    fn test_display_methods() {
        let features = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let labels = Tensor::<f32>::from_vec(vec![0.0, 1.0], &[2]).unwrap();

        let dataset = TensorDataset::new(features, labels);

        // Test display methods don't panic
        let preview = dataset.sample_preview(1).unwrap();
        let preview_text = preview.display();
        assert!(preview_text.contains("Dataset Sample Preview"));

        let distribution = dataset.feature_distribution(None).unwrap();
        let dist_text = distribution.display();
        assert!(dist_text.contains("Dataset Distribution Analysis"));

        let class_dist = dataset.class_distribution().unwrap();
        let class_text = class_dist.display();
        assert!(class_text.contains("Class Distribution"));

        let histogram = dataset.feature_histogram(0, 2).unwrap();
        let hist_text = histogram.display(10);
        assert!(hist_text.contains("Histogram"));
    }

    #[test]
    fn test_augmentation_effects_analysis() {
        use crate::transforms::Transform;

        // Create a simple test transform that adds 1.0 to all features
        struct AddOneTransform;

        impl Transform<f32> for AddOneTransform {
            fn apply(
                &self,
                sample: (tenflowers_core::Tensor<f32>, tenflowers_core::Tensor<f32>),
            ) -> tenflowers_core::Result<(tenflowers_core::Tensor<f32>, tenflowers_core::Tensor<f32>)>
            {
                let (features, labels) = sample;
                if let Some(data) = features.as_slice() {
                    let transformed_data: Vec<f32> = data.iter().map(|x| x + 1.0).collect();
                    let transformed_features = tenflowers_core::Tensor::from_vec(
                        transformed_data,
                        features.shape().dims(),
                    )?;
                    Ok((transformed_features, labels))
                } else {
                    Ok((features, labels))
                }
            }
        }

        let features =
            tenflowers_core::Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2])
                .unwrap();
        let labels = tenflowers_core::Tensor::<f32>::from_vec(vec![0.0, 1.0, 2.0], &[3]).unwrap();

        let dataset = crate::TensorDataset::new(features, labels);
        let transform = AddOneTransform;

        let analysis = dataset.analyze_augmentation_effects(&transform, 2).unwrap();

        assert_eq!(analysis.samples_analyzed, 2);
        assert!(analysis.transform_success_rate > 0.0);
        assert_eq!(analysis.feature_changes.feature_count, 2);
        assert!(analysis.feature_changes.average_change > 0.0);

        // The transform adds 1.0, so mean should increase by 1.0
        let expected_mean_change = 1.0;
        let actual_mean_change = analysis.distribution_changes.mean_change;
        assert!((actual_mean_change - expected_mean_change).abs() < 0.001);
    }

    #[test]
    fn test_sample_comparison() {
        use crate::transforms::Transform;

        // Create a transform that multiplies by 2
        struct MultiplyByTwoTransform;

        impl Transform<f32> for MultiplyByTwoTransform {
            fn apply(
                &self,
                sample: (tenflowers_core::Tensor<f32>, tenflowers_core::Tensor<f32>),
            ) -> tenflowers_core::Result<(tenflowers_core::Tensor<f32>, tenflowers_core::Tensor<f32>)>
            {
                let (features, labels) = sample;
                if let Some(data) = features.as_slice() {
                    let transformed_data: Vec<f32> = data.iter().map(|x| x * 2.0).collect();
                    let transformed_features = tenflowers_core::Tensor::from_vec(
                        transformed_data,
                        features.shape().dims(),
                    )?;
                    Ok((transformed_features, labels))
                } else {
                    Ok((features, labels))
                }
            }
        }

        let sample1 = (
            tenflowers_core::Tensor::<f32>::from_vec(vec![1.0, 2.0], &[2]).unwrap(),
            tenflowers_core::Tensor::<f32>::from_vec(vec![0.0], &[1]).unwrap(),
        );
        let sample2 = (
            tenflowers_core::Tensor::<f32>::from_vec(vec![3.0, 4.0], &[2]).unwrap(),
            tenflowers_core::Tensor::<f32>::from_vec(vec![1.0], &[1]).unwrap(),
        );

        let samples = vec![sample1, sample2];
        let transform = MultiplyByTwoTransform;

        let comparisons = DatasetVisualizer::compare_samples(&samples, &transform, 2).unwrap();

        assert_eq!(comparisons.len(), 2);

        // Check that the change magnitude is non-zero
        for comparison in &comparisons {
            assert!(comparison.change_magnitude > 0.0);
            assert!(comparison.transformed_stats.mean > comparison.original_stats.mean);
        }
    }

    #[test]
    fn test_augmentation_effects_display() {
        use crate::transforms::Transform;

        // Identity transform for testing display
        struct IdentityTransform;

        impl Transform<f32> for IdentityTransform {
            fn apply(
                &self,
                sample: (tenflowers_core::Tensor<f32>, tenflowers_core::Tensor<f32>),
            ) -> tenflowers_core::Result<(tenflowers_core::Tensor<f32>, tenflowers_core::Tensor<f32>)>
            {
                Ok(sample)
            }
        }

        let features =
            tenflowers_core::Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let labels = tenflowers_core::Tensor::<f32>::from_vec(vec![0.0, 1.0], &[2]).unwrap();

        let dataset = crate::TensorDataset::new(features, labels);
        let transform = IdentityTransform;

        let analysis = dataset.analyze_augmentation_effects(&transform, 2).unwrap();
        let display_text = analysis.display();

        assert!(display_text.contains("Augmentation Effects Analysis"));
        assert!(display_text.contains("Samples analyzed"));
        assert!(display_text.contains("Transform success rate"));
        assert!(display_text.contains("Feature Change Analysis"));
        assert!(display_text.contains("Distribution Change Analysis"));
    }
}
