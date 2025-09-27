//! Core DataLoader Functionality
//!
//! This module provides the main DataLoader implementation with multi-threaded
//! batch loading, prefetching capabilities, and NUMA-aware scheduling.

use super::batch_result::BatchResult;
use super::collate::{CollateFn, DefaultCollate};
use super::samplers::Sampler;
use crate::numa_scheduler::NumaConfig;
use crate::Dataset;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::Duration;
use tenflowers_core::{Device, Result, TensorError};

/// Configuration for DataLoader
#[derive(Debug, Clone)]
pub struct DataLoaderConfig {
    pub batch_size: usize,
    pub num_workers: usize,
    pub prefetch_factor: usize,
    pub pin_memory: bool,
    pub drop_last: bool,
    pub timeout: Option<Duration>,
    pub collate_batches: bool,
    pub target_device: Option<Device>,
    /// NUMA-aware scheduling configuration
    pub numa_config: Option<NumaConfig>,
}

impl Default for DataLoaderConfig {
    fn default() -> Self {
        Self {
            batch_size: 1,
            num_workers: 1,
            prefetch_factor: 2,
            pin_memory: false,
            drop_last: false,
            timeout: Some(Duration::from_secs(30)),
            collate_batches: true,
            target_device: None,
            numa_config: None, // NUMA disabled by default for compatibility
        }
    }
}

impl DataLoaderConfig {
    /// Enable NUMA-aware scheduling with default configuration
    pub fn with_numa_scheduling(mut self) -> Self {
        self.numa_config = Some(NumaConfig::default());
        self
    }

    /// Enable NUMA-aware scheduling with custom configuration
    pub fn with_numa_config(mut self, numa_config: NumaConfig) -> Self {
        self.numa_config = Some(numa_config);
        self
    }

    /// Check if NUMA scheduling is enabled
    pub fn is_numa_enabled(&self) -> bool {
        self.numa_config
            .as_ref()
            .is_some_and(|config| config.enabled)
    }
}

/// Multi-threaded data loader with prefetching
pub struct DataLoader<T, D: Dataset<T>, S: Sampler> {
    dataset: Arc<D>,
    sampler: Arc<Mutex<S>>,
    config: DataLoaderConfig,
    _phantom: std::marker::PhantomData<T>,
}

impl<T, D: Dataset<T> + Send + Sync + 'static, S: Sampler + 'static> DataLoader<T, D, S>
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
    pub fn new(dataset: D, sampler: S, config: DataLoaderConfig) -> Self {
        Self {
            dataset: Arc::new(dataset),
            sampler: Arc::new(Mutex::new(sampler)),
            config,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create iterator over batches
    pub fn iter(&self) -> DataLoaderIterator<T, D, S> {
        let dataset_len = self.dataset.len();
        let indices = {
            let sampler = self.sampler.lock().unwrap();
            sampler.sample_indices(dataset_len).collect::<Vec<_>>()
        };

        DataLoaderIterator::new(Arc::clone(&self.dataset), indices, self.config.clone())
    }

    /// Get a reference to the configuration
    pub fn config(&self) -> &DataLoaderConfig {
        &self.config
    }
}

/// Iterator for DataLoader that handles multi-threaded batch loading with prefetching
pub struct DataLoaderIterator<T, D: Dataset<T> + Send + Sync + 'static, S: Sampler> {
    dataset: Arc<D>,
    indices: Vec<usize>,
    config: DataLoaderConfig,
    current_batch: usize,
    total_batches: usize,
    prefetch_queue: Arc<Mutex<VecDeque<Result<BatchResult<T>>>>>,
    worker_handles: Vec<thread::JoinHandle<()>>,
    receiver: Option<mpsc::Receiver<Result<BatchResult<T>>>>,
    prefetcher_handle: Option<thread::JoinHandle<()>>,
    shutdown_signal: Arc<AtomicBool>,
    _phantom: std::marker::PhantomData<S>,
}

impl<T, D: Dataset<T> + Send + Sync + 'static, S: Sampler> DataLoaderIterator<T, D, S>
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
    fn new(dataset: Arc<D>, indices: Vec<usize>, config: DataLoaderConfig) -> Self {
        let total_batches = if config.drop_last {
            indices.len() / config.batch_size
        } else {
            (indices.len() + config.batch_size - 1) / config.batch_size
        };

        let (sender, receiver) = mpsc::channel();
        let prefetch_queue = Arc::new(Mutex::new(VecDeque::new()));
        let shutdown_signal = Arc::new(AtomicBool::new(false));

        // Start worker threads if multi-threading is enabled
        let worker_handles = if config.num_workers > 1 {
            let mut handles = Vec::new();
            let batch_indices = Self::distribute_batches(&indices, &config);

            for worker_indices in batch_indices {
                let dataset_clone = Arc::clone(&dataset);
                let sender_clone = sender.clone();
                let config_clone = config.clone();

                let handle = thread::spawn(move || {
                    for batch_start in (0..worker_indices.len()).step_by(config_clone.batch_size) {
                        let batch_end =
                            (batch_start + config_clone.batch_size).min(worker_indices.len());
                        let batch_indices = &worker_indices[batch_start..batch_end];

                        let batch_result =
                            Self::load_batch(&dataset_clone, batch_indices, &config_clone);
                        if sender_clone.send(batch_result).is_err() {
                            break; // Receiver has been dropped
                        }
                    }
                });

                handles.push(handle);
            }
            handles
        } else {
            Vec::new()
        };

        // Drop the original sender so the receiver will know when all workers are done
        drop(sender);

        // Start prefetcher thread for single-threaded or as additional prefetching for multi-threaded
        let prefetcher_handle = if config.prefetch_factor > 0 {
            let dataset_clone = Arc::clone(&dataset);
            let indices_clone = indices.clone();
            let config_clone = config.clone();
            let prefetch_queue_clone = Arc::clone(&prefetch_queue);
            let shutdown_clone = Arc::clone(&shutdown_signal);

            Some(thread::spawn(move || {
                Self::prefetcher_worker(
                    dataset_clone,
                    indices_clone,
                    config_clone,
                    prefetch_queue_clone,
                    shutdown_clone,
                );
            }))
        } else {
            None
        };

        Self {
            dataset,
            indices,
            config,
            current_batch: 0,
            total_batches,
            prefetch_queue,
            worker_handles,
            receiver: Some(receiver),
            prefetcher_handle,
            shutdown_signal,
            _phantom: std::marker::PhantomData,
        }
    }

    fn distribute_batches(indices: &[usize], config: &DataLoaderConfig) -> Vec<Vec<usize>> {
        let chunk_size = (indices.len() + config.num_workers - 1) / config.num_workers;
        indices
            .chunks(chunk_size)
            .map(|chunk| chunk.to_vec())
            .collect()
    }

    fn load_batch(
        dataset: &Arc<D>,
        batch_indices: &[usize],
        config: &DataLoaderConfig,
    ) -> Result<BatchResult<T>> {
        let mut batch = Vec::with_capacity(batch_indices.len());

        for &idx in batch_indices {
            match dataset.get(idx) {
                Ok(sample) => {
                    // If target device is specified, move tensors to target device
                    if let Some(device) = config.target_device {
                        let (features, labels) = sample;
                        let features_on_device = features.to(device)?;
                        let labels_on_device = labels.to(device)?;
                        batch.push((features_on_device, labels_on_device));
                    } else {
                        batch.push(sample);
                    }
                }
                Err(e) => return Err(e),
            }
        }

        if config.collate_batches && batch.len() > 1 {
            // Collate the batch into stacked tensors
            let collate_fn = DefaultCollate;
            let (features, labels) = collate_fn.collate(batch)?;
            Ok(BatchResult::Collated(features, labels))
        } else {
            // Return individual samples
            Ok(BatchResult::Samples(batch))
        }
    }

    fn load_batch_single_threaded(&self) -> Option<Result<BatchResult<T>>> {
        if self.current_batch >= self.total_batches {
            return None;
        }

        let batch_start = self.current_batch * self.config.batch_size;
        let batch_end = if self.config.drop_last {
            // For drop_last, ensure we have a complete batch
            if batch_start + self.config.batch_size <= self.indices.len() {
                batch_start + self.config.batch_size
            } else {
                batch_start // This will be caught by the condition below
            }
        } else {
            (batch_start + self.config.batch_size).min(self.indices.len())
        };

        if batch_start >= self.indices.len()
            || (self.config.drop_last && batch_end - batch_start < self.config.batch_size)
        {
            return None;
        }

        let batch_indices = &self.indices[batch_start..batch_end];
        Some(Self::load_batch(&self.dataset, batch_indices, &self.config))
    }

    /// Prefetcher worker that loads batches ahead of time
    fn prefetcher_worker(
        dataset: Arc<D>,
        indices: Vec<usize>,
        config: DataLoaderConfig,
        prefetch_queue: Arc<Mutex<VecDeque<Result<BatchResult<T>>>>>,
        shutdown_signal: Arc<AtomicBool>,
    ) {
        let total_batches = if config.drop_last {
            indices.len() / config.batch_size
        } else {
            (indices.len() + config.batch_size - 1) / config.batch_size
        };

        let max_prefetch_size = config.prefetch_factor * config.batch_size;

        for batch_idx in 0..total_batches {
            // Check if we should stop
            if shutdown_signal.load(Ordering::Relaxed) {
                break;
            }

            // Wait if prefetch queue is full
            loop {
                {
                    let queue = prefetch_queue.lock().unwrap();
                    if queue.len() < max_prefetch_size {
                        break;
                    }
                }

                // Check shutdown signal while waiting
                if shutdown_signal.load(Ordering::Relaxed) {
                    return;
                }

                thread::sleep(Duration::from_millis(1));
            }

            // Load the batch
            let batch_start = batch_idx * config.batch_size;
            let batch_end = if config.drop_last {
                // For drop_last, ensure we have a complete batch
                if batch_start + config.batch_size <= indices.len() {
                    batch_start + config.batch_size
                } else {
                    batch_start // This will be caught by the condition below
                }
            } else {
                (batch_start + config.batch_size).min(indices.len())
            };

            if batch_start >= indices.len()
                || (config.drop_last && batch_end - batch_start < config.batch_size)
            {
                break;
            }

            let batch_indices = &indices[batch_start..batch_end];
            let batch_result = Self::load_batch(&dataset, batch_indices, &config);

            // Add to prefetch queue
            {
                let mut queue = prefetch_queue.lock().unwrap();
                queue.push_back(batch_result);
            }
        }
    }
}

impl<T, D: Dataset<T> + Send + Sync + 'static, S: Sampler> Iterator for DataLoaderIterator<T, D, S>
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
    type Item = Result<BatchResult<T>>;

    fn next(&mut self) -> Option<Self::Item> {
        // Check if we've exceeded the total batches
        if self.current_batch >= self.total_batches {
            return None;
        }

        // If prefetching is enabled, use prefetch queue exclusively
        if self.config.prefetch_factor > 0 {
            // For prefetching, wait for batches to be available in the queue
            loop {
                {
                    let mut queue = self.prefetch_queue.lock().unwrap();
                    if let Some(batch_result) = queue.pop_front() {
                        self.current_batch += 1;
                        return Some(batch_result);
                    }
                }

                // Check if we've reached the end
                if self.current_batch >= self.total_batches {
                    return None;
                }

                // Small delay to avoid busy waiting
                std::thread::sleep(Duration::from_millis(1));
            }
        }

        // No prefetching - use direct loading methods
        if self.config.num_workers <= 1 {
            // Single-threaded loading
            let result = self.load_batch_single_threaded();
            if result.is_some() {
                self.current_batch += 1;
            }
            result
        } else {
            // Multi-threaded loading
            if let Some(ref receiver) = self.receiver {
                match receiver.recv_timeout(self.config.timeout.unwrap_or(Duration::from_secs(30)))
                {
                    Ok(batch) => {
                        self.current_batch += 1;
                        Some(batch)
                    }
                    Err(mpsc::RecvTimeoutError::Timeout) => {
                        Some(Err(TensorError::invalid_argument(
                            "DataLoader timeout waiting for batch".to_string(),
                        )))
                    }
                    Err(mpsc::RecvTimeoutError::Disconnected) => None,
                }
            } else {
                None
            }
        }
    }
}

impl<T, D: Dataset<T> + Send + Sync + 'static, S: Sampler> Drop for DataLoaderIterator<T, D, S> {
    fn drop(&mut self) {
        // Signal shutdown to all threads
        self.shutdown_signal.store(true, Ordering::Relaxed);

        // Wait for the prefetcher thread to complete
        if let Some(handle) = self.prefetcher_handle.take() {
            let _ = handle.join();
        }

        // Wait for all worker threads to complete
        for handle in self.worker_handles.drain(..) {
            let _ = handle.join();
        }
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

        fn get(&self, index: usize) -> Result<(Tensor<f32>, Tensor<f32>)> {
            if index < self.size {
                let features = Tensor::ones(&[2]);
                let labels = Tensor::zeros(&[1]);
                Ok((features, labels))
            } else {
                Err(TensorError::invalid_argument(
                    "Index out of bounds".to_string(),
                ))
            }
        }
    }

    #[test]
    fn test_dataloader_config_default() {
        let config = DataLoaderConfig::default();
        assert_eq!(config.batch_size, 1);
        assert_eq!(config.num_workers, 1);
        assert_eq!(config.prefetch_factor, 2);
        assert!(!config.pin_memory);
        assert!(!config.drop_last);
        assert!(config.collate_batches);
        assert!(config.target_device.is_none());
        assert!(!config.is_numa_enabled());
    }

    #[test]
    fn test_dataloader_config_numa() {
        let config = DataLoaderConfig::default().with_numa_scheduling();
        assert!(config.is_numa_enabled());
    }

    #[test]
    fn test_dataloader_creation() {
        let dataset = MockDataset::new(10);
        let sampler = SequentialSampler::new();
        let config = DataLoaderConfig::default();

        let dataloader = DataLoader::new(dataset, sampler, config);
        // Should create without error
        assert_eq!(dataloader.dataset.len(), 10);
    }

    #[test]
    fn test_dataloader_iterator_single_threaded() {
        let dataset = MockDataset::new(5);
        let sampler = SequentialSampler::new();
        let mut config = DataLoaderConfig::default();
        config.batch_size = 2;
        config.num_workers = 1;
        config.prefetch_factor = 0; // Disable prefetching for simpler test

        let dataloader = DataLoader::new(dataset, sampler, config);
        let mut iter = dataloader.iter();

        // Should get 3 batches: [0,1], [2,3], [4]
        let batch1 = iter.next();
        assert!(batch1.is_some());
        assert!(batch1.unwrap().is_ok());

        let batch2 = iter.next();
        assert!(batch2.is_some());
        assert!(batch2.unwrap().is_ok());

        let batch3 = iter.next();
        assert!(batch3.is_some());
        assert!(batch3.unwrap().is_ok());

        let batch4 = iter.next();
        assert!(batch4.is_none());
    }

    #[test]
    fn test_dataloader_drop_last() {
        let dataset = MockDataset::new(5);
        let sampler = SequentialSampler::new();
        let mut config = DataLoaderConfig::default();
        config.batch_size = 2;
        config.drop_last = true;
        config.prefetch_factor = 0; // Disable prefetching

        let dataloader = DataLoader::new(dataset, sampler, config);
        let mut iter = dataloader.iter();

        // Should get 2 complete batches: [0,1], [2,3] (drop [4])
        let batch1 = iter.next();
        assert!(batch1.is_some());

        let batch2 = iter.next();
        assert!(batch2.is_some());

        let batch3 = iter.next();
        assert!(batch3.is_none()); // Third batch is dropped due to incomplete size
    }
}
