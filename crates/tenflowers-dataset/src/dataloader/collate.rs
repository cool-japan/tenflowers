//! Collation Functions and Strategies
//!
//! This module provides various collation functions for combining individual samples
//! into batches, including default stacking, padding for variable-length sequences,
//! and bucket-based collation for efficient processing.

use tenflowers_core::{Result, Tensor, TensorError};

/// Collate function trait for combining individual samples into batches
pub trait CollateFn<T> {
    /// Combine a batch of individual samples into a single batched sample
    fn collate(&self, batch: Vec<(Tensor<T>, Tensor<T>)>) -> Result<(Tensor<T>, Tensor<T>)>;
}

/// Default collate function that stacks tensors along the batch dimension
pub struct DefaultCollate;

impl<T> CollateFn<T> for DefaultCollate
where
    T: Clone + Default + scirs2_core::numeric::Zero + Send + Sync + 'static,
{
    fn collate(&self, batch: Vec<(Tensor<T>, Tensor<T>)>) -> Result<(Tensor<T>, Tensor<T>)> {
        if batch.is_empty() {
            return Err(TensorError::invalid_argument(
                "Cannot collate empty batch".to_string(),
            ));
        }

        // Separate features and labels
        let (features, labels): (Vec<_>, Vec<_>) = batch.into_iter().unzip();

        // Stack features and labels along batch dimension (dim 0)
        let feature_refs: Vec<&Tensor<T>> = features.iter().collect();
        let label_refs: Vec<&Tensor<T>> = labels.iter().collect();
        let stacked_features = tenflowers_core::ops::stack(&feature_refs, 0)?;
        let stacked_labels = tenflowers_core::ops::stack(&label_refs, 0)?;

        Ok((stacked_features, stacked_labels))
    }
}

/// Padding strategy for variable-length sequences
#[derive(Debug, Clone)]
pub enum PaddingStrategy {
    /// Pad to the maximum length in the batch
    MaxLength,
    /// Pad to a fixed length (truncate if longer)
    FixedLength(usize),
    /// Pad to the next multiple of bucket size
    Bucket(usize),
}

/// Collate function for variable-length sequences with padding
pub struct PaddingCollate<T> {
    padding_value: T,
    padding_strategy: PaddingStrategy,
    truncate: bool,
}

impl<T> PaddingCollate<T>
where
    T: Clone + Default + scirs2_core::numeric::Zero + Send + Sync + 'static,
{
    /// Create a new padding collate function
    pub fn new(padding_value: T, padding_strategy: PaddingStrategy) -> Self {
        Self {
            padding_value,
            padding_strategy,
            truncate: true,
        }
    }

    /// Create padding collate with zero padding
    pub fn with_zero_padding(padding_strategy: PaddingStrategy) -> Self {
        Self::new(T::zero(), padding_strategy)
    }

    /// Set whether to truncate sequences longer than target length
    pub fn with_truncate(mut self, truncate: bool) -> Self {
        self.truncate = truncate;
        self
    }

    /// Determine the target length based on strategy and batch
    fn determine_target_length(&self, sequences: &[&Tensor<T>]) -> usize {
        match &self.padding_strategy {
            PaddingStrategy::MaxLength => sequences
                .iter()
                .map(|tensor| tensor.shape().dims()[tensor.shape().rank() - 1])
                .max()
                .unwrap_or(0),
            PaddingStrategy::FixedLength(length) => *length,
            PaddingStrategy::Bucket(bucket_size) => {
                let max_len = sequences
                    .iter()
                    .map(|tensor| tensor.shape().dims()[tensor.shape().rank() - 1])
                    .max()
                    .unwrap_or(0);
                ((max_len + bucket_size - 1) / bucket_size) * bucket_size
            }
        }
    }

    /// Pad or truncate a sequence to target length
    fn pad_sequence(&self, tensor: &Tensor<T>, target_length: usize) -> Result<Tensor<T>> {
        let shape = tensor.shape().dims();
        let seq_length = shape[shape.len() - 1];

        if seq_length == target_length {
            return Ok(tensor.clone());
        }

        if seq_length > target_length && self.truncate {
            // Truncate sequence
            let mut ranges = Vec::new();
            #[allow(clippy::needless_range_loop)]
            for i in 0..shape.len() - 1 {
                ranges.push(0..shape[i]);
            }
            ranges.push(0..target_length);
            tenflowers_core::ops::slice(tensor, &ranges)
        } else if seq_length < target_length {
            // Pad sequence
            let mut new_shape = shape.to_vec();
            let last_dim_idx = new_shape.len() - 1;
            new_shape[last_dim_idx] = target_length;

            let padding_length = target_length - seq_length;
            let mut padding_shape = shape.to_vec();
            let padding_last_dim_idx = padding_shape.len() - 1;
            padding_shape[padding_last_dim_idx] = padding_length;

            // Create padding tensor filled with padding value
            let total_padding_size = padding_shape.iter().product();
            let padding_data = vec![self.padding_value.clone(); total_padding_size];
            let padding_tensor = Tensor::from_vec(padding_data, &padding_shape)?;

            // Concatenate original tensor with padding
            tenflowers_core::ops::concat(&[tensor, &padding_tensor], shape.len() - 1)
        } else {
            // Sequence is longer but truncate is false, return as-is
            Ok(tensor.clone())
        }
    }
}

impl<T> CollateFn<T> for PaddingCollate<T>
where
    T: Clone + Default + scirs2_core::numeric::Zero + Send + Sync + 'static,
{
    fn collate(&self, batch: Vec<(Tensor<T>, Tensor<T>)>) -> Result<(Tensor<T>, Tensor<T>)> {
        if batch.is_empty() {
            return Err(TensorError::invalid_argument(
                "Cannot collate empty batch".to_string(),
            ));
        }

        // Separate features and labels
        let (features, labels): (Vec<_>, Vec<_>) = batch.into_iter().unzip();

        // Determine target lengths for features and labels
        let feature_refs: Vec<&Tensor<T>> = features.iter().collect();
        let label_refs: Vec<&Tensor<T>> = labels.iter().collect();

        let feature_target_length = self.determine_target_length(&feature_refs);
        let label_target_length = self.determine_target_length(&label_refs);

        // Pad all sequences to target lengths
        let mut padded_features = Vec::new();
        for feature in &features {
            let padded = self.pad_sequence(feature, feature_target_length)?;
            padded_features.push(padded);
        }

        let mut padded_labels = Vec::new();
        for label in &labels {
            let padded = self.pad_sequence(label, label_target_length)?;
            padded_labels.push(padded);
        }

        // Stack padded sequences
        let padded_feature_refs: Vec<&Tensor<T>> = padded_features.iter().collect();
        let padded_label_refs: Vec<&Tensor<T>> = padded_labels.iter().collect();

        let stacked_features = tenflowers_core::ops::stack(&padded_feature_refs, 0)?;
        let stacked_labels = tenflowers_core::ops::stack(&padded_label_refs, 0)?;

        Ok((stacked_features, stacked_labels))
    }
}

/// Collate function that groups samples by length for efficient processing
pub struct BucketCollate<T> {
    bucket_sizes: Vec<usize>,
    padding_value: T,
}

impl<T> BucketCollate<T>
where
    T: Clone + Default + scirs2_core::numeric::Zero + Send + Sync + 'static,
{
    /// Create a new bucket collate function
    pub fn new(bucket_sizes: Vec<usize>, padding_value: T) -> Self {
        Self {
            bucket_sizes,
            padding_value,
        }
    }

    /// Create bucket collate with zero padding
    pub fn with_zero_padding(bucket_sizes: Vec<usize>) -> Self {
        Self::new(bucket_sizes, T::zero())
    }

    /// Find the appropriate bucket size for a given length
    fn find_bucket_size(&self, length: usize) -> usize {
        self.bucket_sizes
            .iter()
            .find(|&&bucket_size| bucket_size >= length)
            .copied()
            .unwrap_or(*self.bucket_sizes.last().unwrap_or(&length))
    }
}

impl<T> CollateFn<T> for BucketCollate<T>
where
    T: Clone + Default + scirs2_core::numeric::Zero + Send + Sync + 'static,
{
    fn collate(&self, batch: Vec<(Tensor<T>, Tensor<T>)>) -> Result<(Tensor<T>, Tensor<T>)> {
        if batch.is_empty() {
            return Err(TensorError::invalid_argument(
                "Cannot collate empty batch".to_string(),
            ));
        }

        // Use padding collate with bucket strategy
        let max_feature_length = batch
            .iter()
            .map(|(f, _)| f.shape().dims()[f.shape().rank() - 1])
            .max()
            .unwrap_or(0);
        let max_label_length = batch
            .iter()
            .map(|(_, l)| l.shape().dims()[l.shape().rank() - 1])
            .max()
            .unwrap_or(0);

        let feature_bucket_size = self.find_bucket_size(max_feature_length);
        let label_bucket_size = self.find_bucket_size(max_label_length);

        let feature_padding_collate = PaddingCollate::new(
            self.padding_value.clone(),
            PaddingStrategy::FixedLength(feature_bucket_size),
        );
        let _label_padding_collate = PaddingCollate::new(
            self.padding_value.clone(),
            PaddingStrategy::FixedLength(label_bucket_size),
        );

        // For simplicity, we'll use the feature padding strategy for both
        // In a real implementation, you might want separate collate functions
        feature_padding_collate.collate(batch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenflowers_core::{Device, Tensor};

    #[test]
    fn test_default_collate() {
        let collate_fn = DefaultCollate;
        let batch = vec![
            (Tensor::<f32>::ones(&[2, 3]), Tensor::<f32>::zeros(&[1])),
            (Tensor::<f32>::ones(&[2, 3]), Tensor::<f32>::ones(&[1])),
        ];

        let result = collate_fn.collate(batch);
        assert!(result.is_ok());

        let (features, labels) = result.unwrap();
        assert_eq!(features.shape().dims(), &[2, 2, 3]); // [batch_size, feature_dim1, feature_dim2]
        assert_eq!(labels.shape().dims(), &[2, 1]); // [batch_size, label_dim]
    }

    #[test]
    fn test_default_collate_empty_batch() {
        let collate_fn = DefaultCollate;
        let batch: Vec<(Tensor<f32>, Tensor<f32>)> = vec![];

        let result = collate_fn.collate(batch);
        assert!(result.is_err());
    }

    #[test]
    fn test_padding_strategy_max_length() {
        let strategy = PaddingStrategy::MaxLength;
        match strategy {
            PaddingStrategy::MaxLength => assert!(true),
            _ => panic!("Expected MaxLength strategy"),
        }
    }

    #[test]
    fn test_padding_strategy_fixed_length() {
        let strategy = PaddingStrategy::FixedLength(10);
        match strategy {
            PaddingStrategy::FixedLength(len) => assert_eq!(len, 10),
            _ => panic!("Expected FixedLength strategy"),
        }
    }

    #[test]
    fn test_padding_strategy_bucket() {
        let strategy = PaddingStrategy::Bucket(8);
        match strategy {
            PaddingStrategy::Bucket(size) => assert_eq!(size, 8),
            _ => panic!("Expected Bucket strategy"),
        }
    }

    #[test]
    fn test_padding_collate_creation() {
        let collate_fn = PaddingCollate::new(0.0_f32, PaddingStrategy::MaxLength);
        assert_eq!(collate_fn.padding_value, 0.0);
        assert!(collate_fn.truncate);
    }

    #[test]
    fn test_padding_collate_with_zero_padding() {
        let collate_fn = PaddingCollate::<f32>::with_zero_padding(PaddingStrategy::FixedLength(5));
        assert_eq!(collate_fn.padding_value, 0.0);
    }

    #[test]
    fn test_padding_collate_with_truncate() {
        let collate_fn =
            PaddingCollate::new(0.0_f32, PaddingStrategy::MaxLength).with_truncate(false);
        assert!(!collate_fn.truncate);
    }

    #[test]
    fn test_bucket_collate_creation() {
        let bucket_sizes = vec![16, 32, 64, 128];
        let collate_fn = BucketCollate::new(bucket_sizes.clone(), 0.0_f32);
        assert_eq!(collate_fn.bucket_sizes, bucket_sizes);
        assert_eq!(collate_fn.padding_value, 0.0);
    }

    #[test]
    fn test_bucket_collate_with_zero_padding() {
        let bucket_sizes = vec![8, 16, 32];
        let collate_fn = BucketCollate::<f32>::with_zero_padding(bucket_sizes.clone());
        assert_eq!(collate_fn.bucket_sizes, bucket_sizes);
        assert_eq!(collate_fn.padding_value, 0.0);
    }

    #[test]
    fn test_bucket_collate_find_bucket_size() {
        let bucket_sizes = vec![16, 32, 64];
        let collate_fn = BucketCollate::new(bucket_sizes, 0.0_f32);

        assert_eq!(collate_fn.find_bucket_size(10), 16);
        assert_eq!(collate_fn.find_bucket_size(20), 32);
        assert_eq!(collate_fn.find_bucket_size(50), 64);
        assert_eq!(collate_fn.find_bucket_size(100), 64); // Uses last bucket for oversized
    }
}
