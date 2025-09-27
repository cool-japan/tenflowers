//! DataLoader Builder Pattern
//!
//! This module provides a fluent builder API for creating DataLoader instances
//! with customizable configuration options including batch size, number of workers,
//! device placement, and other settings.

use super::core::{DataLoader, DataLoaderConfig};
use super::samplers::Sampler;
use crate::Dataset;
use std::time::Duration;
use tenflowers_core::Device;

/// Builder for creating DataLoader with fluent API
pub struct DataLoaderBuilder<T, D: Dataset<T>> {
    dataset: D,
    config: DataLoaderConfig,
    _phantom: std::marker::PhantomData<T>,
}

impl<T, D: Dataset<T>> DataLoaderBuilder<T, D>
where
    T: Clone
        + Default
        + num_traits::Zero
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    /// Create a new DataLoader builder with the provided dataset
    pub fn new(dataset: D) -> Self {
        Self {
            dataset,
            config: DataLoaderConfig::default(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Set the batch size for data loading
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.config.batch_size = batch_size;
        self
    }

    /// Set the number of worker threads for parallel data loading
    pub fn num_workers(mut self, num_workers: usize) -> Self {
        self.config.num_workers = num_workers;
        self
    }

    /// Set the prefetch factor for asynchronous batch loading
    pub fn prefetch_factor(mut self, prefetch_factor: usize) -> Self {
        self.config.prefetch_factor = prefetch_factor;
        self
    }

    /// Set the target device for GPU direct loading
    /// If set, tensors will be moved to the specified device during loading
    pub fn target_device(mut self, device: Device) -> Self {
        self.config.target_device = Some(device);
        self
    }

    /// Enable GPU direct loading to the default GPU device
    #[cfg(feature = "gpu")]
    pub fn gpu_direct(mut self) -> Self {
        // Set GPU device directly for reliable behavior
        self.config.target_device = Some(Device::Gpu(0));
        self
    }

    /// Set whether to drop the last incomplete batch
    pub fn drop_last(mut self, drop_last: bool) -> Self {
        self.config.drop_last = drop_last;
        self
    }

    /// Set the timeout for batch loading operations
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.config.timeout = Some(timeout);
        self
    }

    /// Set whether to automatically collate batches into stacked tensors
    pub fn collate_batches(mut self, collate: bool) -> Self {
        self.config.collate_batches = collate;
        self
    }

    /// Enable memory pinning for faster GPU transfers
    pub fn pin_memory(mut self, pin_memory: bool) -> Self {
        self.config.pin_memory = pin_memory;
        self
    }

    /// Build the DataLoader with the specified sampler
    pub fn build<S: Sampler + 'static>(self, sampler: S) -> DataLoader<T, D, S>
    where
        D: Send + Sync + 'static,
        T: Clone + Send + Sync + 'static,
    {
        DataLoader::new(self.dataset, sampler, self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::super::samplers::SequentialSampler;
    use super::*;
    use crate::Dataset;
    use tenflowers_core::{Device, Tensor};

    // Mock dataset for testing
    struct MockDataset {
        size: usize,
    }

    impl MockDataset {
        fn new(size: usize) -> Self {
            Self { size }
        }
    }

    impl Dataset<f32> for MockDataset {
        fn len(&self) -> usize {
            self.size
        }

        fn get(&self, index: usize) -> tenflowers_core::Result<(Tensor<f32>, Tensor<f32>)> {
            if index < self.size {
                let features = Tensor::ones(&[2]);
                let labels = Tensor::zeros(&[1]);
                Ok((features, labels))
            } else {
                Err(tenflowers_core::TensorError::invalid_argument(
                    "Index out of bounds".to_string(),
                ))
            }
        }
    }

    #[test]
    fn test_builder_default_config() {
        let dataset = MockDataset::new(10);
        let sampler = SequentialSampler::new();

        let dataloader = DataLoaderBuilder::new(dataset).build(sampler);

        // Test that default config is applied
        let config = dataloader.config();
        assert_eq!(config.batch_size, 1);
        assert_eq!(config.num_workers, 1);
        assert_eq!(config.prefetch_factor, 2);
        assert!(!config.pin_memory);
        assert!(!config.drop_last);
        assert!(config.collate_batches);
    }

    #[test]
    fn test_builder_custom_config() {
        let dataset = MockDataset::new(10);
        let sampler = SequentialSampler::new();

        let dataloader = DataLoaderBuilder::new(dataset)
            .batch_size(8)
            .num_workers(4)
            .prefetch_factor(3)
            .drop_last(true)
            .pin_memory(true)
            .collate_batches(false)
            .target_device(Device::Cpu)
            .timeout(Duration::from_secs(60))
            .build(sampler);

        let config = dataloader.config();
        assert_eq!(config.batch_size, 8);
        assert_eq!(config.num_workers, 4);
        assert_eq!(config.prefetch_factor, 3);
        assert!(config.pin_memory);
        assert!(config.drop_last);
        assert!(!config.collate_batches);
        assert_eq!(config.target_device, Some(Device::Cpu));
        assert_eq!(config.timeout, Some(Duration::from_secs(60)));
    }

    #[test]
    fn test_builder_target_device() {
        let dataset = MockDataset::new(5);
        let sampler = SequentialSampler::new();

        let dataloader = DataLoaderBuilder::new(dataset)
            .batch_size(2)
            .target_device(Device::Cpu)
            .build(sampler);

        let config = dataloader.config();
        assert_eq!(config.target_device, Some(Device::Cpu));
    }

    #[test]
    #[cfg(feature = "gpu")]
    fn test_builder_gpu_direct() {
        let dataset = MockDataset::new(5);
        let sampler = SequentialSampler::new();

        let dataloader = DataLoaderBuilder::new(dataset)
            .batch_size(2)
            .gpu_direct()
            .build(sampler);

        let config = dataloader.config();
        // GPU device should be set if available
        assert!(config.target_device.is_some());
    }

    #[test]
    fn test_builder_method_chaining() {
        let dataset = MockDataset::new(10);
        let sampler = SequentialSampler::new();

        // Test that all methods can be chained fluently
        let _dataloader = DataLoaderBuilder::new(dataset)
            .batch_size(4)
            .num_workers(2)
            .prefetch_factor(1)
            .drop_last(false)
            .pin_memory(false)
            .collate_batches(true)
            .timeout(Duration::from_secs(30))
            .build(sampler);

        // If we reach here, the chaining worked
        assert!(true);
    }

    #[test]
    fn test_builder_functional_dataloader() {
        let dataset = MockDataset::new(6);
        let sampler = SequentialSampler::new();

        let dataloader = DataLoaderBuilder::new(dataset)
            .batch_size(2)
            .num_workers(1)
            .prefetch_factor(0) // Disable prefetching for deterministic test
            .build(sampler);

        let mut batch_count = 0;
        for batch_result in dataloader.iter() {
            let batch = batch_result.unwrap();
            assert_eq!(batch.len(), 2);
            batch_count += 1;
        }
        assert_eq!(batch_count, 3); // 6 samples / 2 batch_size = 3 batches
    }
}
