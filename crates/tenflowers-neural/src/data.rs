//! Data pipeline integration for training neural networks
//!
//! This module provides integration between tenflowers-dataset and tenflowers-neural
//! for seamless batch generation, data augmentation, and multi-threaded loading.

use std::marker::PhantomData;
use tenflowers_core::{Device, Result, Tensor};
use tenflowers_dataset::{
    DataLoader, DataLoaderBuilder, Dataset, Normalize, RandomSampler, SequentialSampler,
    TextConfig, Vocabulary,
};

/// Configuration for neural network data pipeline
#[derive(Debug, Clone)]
pub struct DataPipelineConfig {
    pub batch_size: usize,
    pub shuffle: bool,
    pub num_workers: usize,
    pub prefetch_factor: usize,
    pub drop_last: bool,
    pub target_device: Device,
}

impl Default for DataPipelineConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            shuffle: true,
            num_workers: 4,
            prefetch_factor: 2,
            drop_last: true,
            target_device: Device::Cpu,
        }
    }
}

/// Neural network data pipeline that integrates dataset loading with training
pub struct NeuralDataPipeline<T, D>
where
    T: Clone + Send + Sync + 'static,
    D: Dataset<T> + Send + Sync + 'static,
{
    dataset: D,
    config: DataPipelineConfig,
    _phantom: PhantomData<T>,
}

impl<T, D> NeuralDataPipeline<T, D>
where
    T: Clone + Send + Sync + 'static,
    D: Dataset<T> + Send + Sync + 'static,
{
    /// Create a new neural data pipeline
    pub fn new(dataset: D, config: DataPipelineConfig) -> Self {
        Self {
            dataset,
            config,
            _phantom: PhantomData,
        }
    }

    /// Create a data loader for training (consumes the pipeline)
    pub fn into_train_loader(self) -> Result<DataLoader<T, D, RandomSampler>>
    where
        T: Clone
            + Default
            + scirs2_core::num_traits::Zero
            + Send
            + Sync
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        let sampler = RandomSampler::new();

        let loader = DataLoaderBuilder::new(self.dataset)
            .batch_size(self.config.batch_size)
            .num_workers(self.config.num_workers)
            .prefetch_factor(self.config.prefetch_factor)
            .drop_last(self.config.drop_last)
            .build(sampler);

        Ok(loader)
    }

    /// Create a data loader for validation/testing (no shuffling, consumes the pipeline)
    pub fn into_val_loader(self) -> Result<DataLoader<T, D, SequentialSampler>>
    where
        T: Clone
            + Default
            + scirs2_core::num_traits::Zero
            + Send
            + Sync
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        let sampler = SequentialSampler::new();

        let loader = DataLoaderBuilder::new(self.dataset)
            .batch_size(self.config.batch_size)
            .num_workers(self.config.num_workers)
            .prefetch_factor(self.config.prefetch_factor)
            .drop_last(false) // Don't drop last batch for validation
            .build(sampler);

        Ok(loader)
    }
}

/// Simple text classification transform wrapper
#[derive(Debug, Clone)]
pub struct TextClassificationTransform {
    tokenizer: Vocabulary,
    max_length: usize,
}

impl TextClassificationTransform {
    pub fn new(vocab_size: usize, max_length: usize) -> Self {
        // Create a basic vocabulary with special tokens
        let mut config = TextConfig {
            max_vocab_size: Some(vocab_size),
            max_sequence_length: max_length,
            ..Default::default()
        };

        // Create empty vocabulary - in practice this would be built from training data
        let empty_texts = vec![];
        let tokenizer = Vocabulary::from_texts(&empty_texts, &config);

        Self {
            tokenizer,
            max_length,
        }
    }

    pub fn tokenizer(&self) -> &Vocabulary {
        &self.tokenizer
    }
}

/// Common data augmentation transforms for neural networks
pub struct NeuralTransforms;

impl NeuralTransforms {
    /// Standard image classification transforms
    pub fn image_classification(mean: Vec<f32>, std: Vec<f32>) -> Result<Normalize<f32>> {
        Normalize::new(mean, std)
    }

    /// Text classification transforms with vocabulary
    pub fn text_classification(
        vocab_size: usize,
        max_length: usize,
    ) -> Result<TextClassificationTransform> {
        Ok(TextClassificationTransform::new(vocab_size, max_length))
    }
}

/// Training batch iterator that works with the neural network trainer
#[derive(Debug, Clone)]
pub struct TrainingBatch {
    pub features: Tensor<f32>,
    pub labels: Tensor<f32>,
    pub batch_size: usize,
}

impl TrainingBatch {
    /// Create a new training batch
    pub fn new(features: Tensor<f32>, labels: Tensor<f32>) -> Result<Self> {
        let batch_size = features.shape().dims()[0];
        Ok(Self {
            features,
            labels,
            batch_size,
        })
    }

    /// Move batch to specified device
    pub fn to_device(&self, device: &Device) -> Result<Self> {
        let transferred_features = self.features.to_device(device.clone())?;
        let transferred_labels = self.labels.to_device(device.clone())?;

        Ok(Self {
            features: transferred_features,
            labels: transferred_labels,
            batch_size: self.batch_size,
        })
    }
}

/// Helper trait to convert BatchResult to neural network training data
pub trait ToTrainingBatch {
    fn to_training_batch(self) -> Result<TrainingBatch>;
}

impl ToTrainingBatch for (Tensor<f32>, Tensor<f32>) {
    fn to_training_batch(self) -> Result<TrainingBatch> {
        TrainingBatch::new(self.0, self.1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenflowers_dataset::TensorDataset;

    #[test]
    fn test_data_pipeline_creation() {
        // Create a simple tensor dataset
        let features = Tensor::<f32>::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[3, 2], // 3 samples, 2 features each
        )
        .unwrap();
        let labels = Tensor::<f32>::from_vec(
            vec![0.0, 1.0, 0.0],
            &[3], // 3 labels
        )
        .unwrap();

        let dataset = TensorDataset::new(features, labels);

        let config = DataPipelineConfig {
            batch_size: 2,
            shuffle: false,
            num_workers: 1,
            prefetch_factor: 1,
            drop_last: false,
            target_device: Device::Cpu,
        };

        let pipeline = NeuralDataPipeline::new(dataset, config.clone());

        // Test that we can create loaders (must consume pipeline)
        let pipeline2 = NeuralDataPipeline::new(
            TensorDataset::new(
                Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap(),
                Tensor::<f32>::from_vec(vec![0.0, 1.0, 0.0], &[3]).unwrap(),
            ),
            config,
        );

        // Test creation (pipeline is consumed, so we need separate instances)
        assert!(pipeline.into_train_loader().is_ok());
        assert!(pipeline2.into_val_loader().is_ok());
    }

    #[test]
    fn test_training_batch_creation() {
        let features = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let labels = Tensor::from_vec(vec![0.0, 1.0], &[2]).unwrap();

        let batch = TrainingBatch::new(features, labels).unwrap();
        assert_eq!(batch.batch_size, 2);
    }

    #[test]
    fn test_neural_transforms() {
        // Test that transforms can be created
        let normalize = NeuralTransforms::image_classification(
            vec![0.485, 0.456, 0.406],
            vec![0.229, 0.224, 0.225],
        );
        assert!(normalize.is_ok());
    }

    #[test]
    fn test_training_batch_device_transfer() {
        let features = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let labels = Tensor::from_vec(vec![0.0, 1.0], &[2]).unwrap();

        let batch = TrainingBatch::new(features, labels).unwrap();

        // Test CPU to CPU transfer (should work)
        let cpu_batch = batch.to_device(&Device::Cpu).unwrap();
        assert_eq!(cpu_batch.batch_size, 2);
        assert_eq!(cpu_batch.features.device(), &Device::Cpu);
        assert_eq!(cpu_batch.labels.device(), &Device::Cpu);

        // Test that features and labels are actually transferred
        assert_eq!(
            cpu_batch.features.as_slice().unwrap(),
            &[1.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(cpu_batch.labels.as_slice().unwrap(), &[0.0, 1.0]);

        // GPU transfer test is conditional on GPU availability
        #[cfg(feature = "gpu")]
        {
            if let Ok(gpu_device) = Device::try_gpu(0) {
                let gpu_batch = batch.to_device(&gpu_device).unwrap();
                assert_eq!(gpu_batch.batch_size, 2);
                assert_eq!(gpu_batch.features.device(), &gpu_device);
                assert_eq!(gpu_batch.labels.device(), &gpu_device);
            }
        }
    }

    #[test]
    fn test_text_classification_transform_creation() {
        // Test the updated text classification transform
        let text_transform = NeuralTransforms::text_classification(1000, 128);
        assert!(text_transform.is_ok());

        let transform = text_transform.unwrap();
        assert_eq!(transform.tokenizer().len(), 4); // Should start with 4 special tokens
    }
}
