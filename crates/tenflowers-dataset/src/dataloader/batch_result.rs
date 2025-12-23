//! Batch Result Types and Operations
//!
//! This module provides batch result types that can handle both individual samples
//! and collated (stacked) tensor batches, with conversion utilities between formats.

use super::collate::CollateFn;
use tenflowers_core::{Result, Tensor};

/// Batch result that can be either collated (stacked tensors) or individual samples
#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
pub enum BatchResult<T> {
    /// Individual samples as a vector
    Samples(Vec<(Tensor<T>, Tensor<T>)>),
    /// Collated batch with stacked tensors
    Collated(Tensor<T>, Tensor<T>),
}

impl<T> BatchResult<T>
where
    T: Clone + Default + scirs2_core::numeric::Zero + Send + Sync + 'static,
{
    /// Get the batch size (number of samples)
    pub fn len(&self) -> usize {
        match self {
            BatchResult::Samples(samples) => samples.len(),
            BatchResult::Collated(features, _) => features.shape().dims()[0],
        }
    }

    /// Check if the batch is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Convert to individual samples if not already
    pub fn into_samples(self) -> Result<Vec<(Tensor<T>, Tensor<T>)>>
    where
        T: Clone + Default + scirs2_core::numeric::Zero + Send + Sync + 'static,
    {
        match self {
            BatchResult::Samples(samples) => Ok(samples),
            BatchResult::Collated(features, labels) => {
                // Unstack the batch back into individual samples
                let batch_size = features.shape().dims()[0];
                let mut samples = Vec::with_capacity(batch_size);

                for i in 0..batch_size {
                    #[allow(clippy::single_range_in_vec_init)]
                    let feature_slice = tenflowers_core::ops::slice(&features, &[i..i + 1])?;
                    #[allow(clippy::single_range_in_vec_init)]
                    let label_slice = tenflowers_core::ops::slice(&labels, &[i..i + 1])?;

                    // Squeeze the first dimension
                    let squeezed_feature = if feature_slice.shape().rank() > 1 {
                        let new_shape: Vec<usize> = feature_slice.shape().dims()[1..].to_vec();
                        tenflowers_core::ops::reshape(&feature_slice, &new_shape)?
                    } else {
                        feature_slice
                    };

                    let squeezed_label = if label_slice.shape().rank() > 1 {
                        let new_shape: Vec<usize> = label_slice.shape().dims()[1..].to_vec();
                        tenflowers_core::ops::reshape(&label_slice, &new_shape)?
                    } else {
                        label_slice
                    };

                    samples.push((squeezed_feature, squeezed_label));
                }

                Ok(samples)
            }
        }
    }

    /// Convert to collated batch if not already
    pub fn into_collated(self) -> Result<(Tensor<T>, Tensor<T>)>
    where
        T: Clone + Default + scirs2_core::numeric::Zero + Send + Sync + 'static,
    {
        match self {
            BatchResult::Samples(samples) => {
                let collate_fn = super::collate::DefaultCollate;
                collate_fn.collate(samples)
            }
            BatchResult::Collated(features, labels) => Ok((features, labels)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenflowers_core::{Device, Tensor};

    #[test]
    fn test_batch_result_len() {
        // Test with samples
        let samples = vec![
            (Tensor::<f32>::ones(&[2, 3]), Tensor::<f32>::zeros(&[1])),
            (Tensor::<f32>::ones(&[2, 3]), Tensor::<f32>::zeros(&[1])),
        ];
        let batch_samples = BatchResult::Samples(samples);
        assert_eq!(batch_samples.len(), 2);

        // Test with collated
        let features = Tensor::<f32>::ones(&[3, 2, 3]);
        let labels = Tensor::<f32>::zeros(&[3, 1]);
        let batch_collated = BatchResult::Collated(features, labels);
        assert_eq!(batch_collated.len(), 3);
    }

    #[test]
    fn test_batch_result_is_empty() {
        let empty_samples = BatchResult::<f32>::Samples(vec![]);
        assert!(empty_samples.is_empty());

        let non_empty_samples = vec![(Tensor::<f32>::ones(&[2]), Tensor::<f32>::zeros(&[1]))];
        let batch = BatchResult::Samples(non_empty_samples);
        assert!(!batch.is_empty());
    }

    #[test]
    fn test_batch_result_samples_variant() {
        let samples = vec![
            (Tensor::<f32>::ones(&[2]), Tensor::<f32>::zeros(&[1])),
            (Tensor::<f32>::ones(&[2]), Tensor::<f32>::ones(&[1])),
        ];
        let batch = BatchResult::Samples(samples.clone());

        match batch {
            BatchResult::Samples(s) => assert_eq!(s.len(), 2),
            _ => panic!("Expected Samples variant"),
        }
    }

    #[test]
    fn test_batch_result_collated_variant() {
        let features: Tensor<f32> = Tensor::ones(&[2, 3]);
        let labels = Tensor::zeros(&[2, 1]);
        let batch = BatchResult::Collated(features.clone(), labels.clone());

        match batch {
            BatchResult::Collated(f, l) => {
                assert_eq!(f.shape().dims(), features.shape().dims());
                assert_eq!(l.shape().dims(), labels.shape().dims());
            }
            _ => panic!("Expected Collated variant"),
        }
    }
}
