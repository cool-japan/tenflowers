//! Enhanced DataLoader with work stealing queue for improved multi-threaded performance
//!
//! This module provides an enhanced version of the DataLoader that uses a work-stealing
//! queue for better load balancing and higher throughput in multi-threaded scenarios.

use crate::dataloader::{BatchResult, CollateFn, DataLoaderConfig, DefaultCollate, Sampler};
use crate::numa_scheduler::{NumaScheduler, NumaWorkerAssignment};
use crate::{Dataset, WorkStealingQueue};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};
use tenflowers_core::{Device, Result, TensorError};

/// Task for workers to execute
#[derive(Debug, Clone)]
struct WorkTask {
    /// Batch of indices to process
    batch_indices: Vec<usize>,
    /// Task ID for ordering
    task_id: usize,
}

/// Result of processing a work task
#[derive(Debug)]
#[allow(dead_code)]
struct WorkResult<T> {
    /// Task ID for ordering
    task_id: usize,
    /// Processed batch result
    batch_result: Result<BatchResult<T>>,
    /// Processing time for performance monitoring
    processing_time: Duration,
}

/// Enhanced DataLoader with work stealing queue
#[allow(dead_code)]
pub struct EnhancedDataLoader<T, D: Dataset<T>> {
    /// Reference to the dataset
    dataset: Arc<D>,
    /// Configuration options
    config: DataLoaderConfig,
    /// Work stealing queue for distributing tasks
    work_queue: Arc<WorkStealingQueue<WorkTask>>,
    /// Results queue for collecting processed batches
    results: Arc<std::sync::Mutex<std::collections::BTreeMap<usize, WorkResult<T>>>>,
    /// Worker thread handles
    worker_handles: Vec<JoinHandle<WorkerStats>>,
    /// Shutdown signal for workers
    shutdown_signal: Arc<AtomicBool>,
    /// Next task ID counter
    next_task_id: AtomicUsize,
    /// Next result ID to return
    next_result_id: AtomicUsize,
    /// Total number of tasks to process
    total_tasks: usize,
    /// Performance statistics
    stats: Arc<std::sync::Mutex<LoaderStats>>,
    /// NUMA scheduler for worker affinity
    numa_scheduler: Option<NumaScheduler>,
    /// NUMA worker assignments
    numa_assignments: Vec<NumaWorkerAssignment>,
}

/// Worker thread statistics
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct WorkerStats {
    /// Worker ID
    worker_id: usize,
    /// Number of tasks processed by this worker
    tasks_processed: usize,
    /// Number of tasks stolen from other workers
    tasks_stolen: usize,
    /// Total processing time
    total_processing_time: Duration,
    /// Number of cache hits (if using caching)
    cache_hits: usize,
}

/// Overall loader statistics
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct LoaderStats {
    /// Total number of batches processed
    batches_processed: usize,
    /// Total processing time across all workers
    total_processing_time: Duration,
    /// Average batch processing time
    average_batch_time: Duration,
    /// Peak memory usage (if available)
    peak_memory_usage: Option<usize>,
    /// Number of work stealing events
    work_stealing_events: usize,
}

impl<T, D: Dataset<T> + Send + Sync + 'static> EnhancedDataLoader<T, D>
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
    /// Create a new enhanced DataLoader with work stealing
    pub fn new<S: Sampler>(dataset: D, config: DataLoaderConfig, sampler: S) -> Result<Self> {
        let dataset = Arc::new(dataset);
        let indices: Vec<usize> = sampler.sample_indices(dataset.len()).collect();

        if indices.is_empty() {
            return Err(TensorError::invalid_argument(
                "No indices to sample".to_string(),
            ));
        }

        // Calculate total number of tasks (batches)
        let total_tasks = if config.drop_last {
            indices.len() / config.batch_size
        } else {
            (indices.len() + config.batch_size - 1) / config.batch_size
        };

        // Create work stealing queue
        let work_queue = Arc::new(WorkStealingQueue::new(config.num_workers));

        // Create results collection
        let results = Arc::new(std::sync::Mutex::new(std::collections::BTreeMap::new()));

        // Initialize shutdown signal and counters
        let shutdown_signal = Arc::new(AtomicBool::new(false));
        let next_task_id = AtomicUsize::new(0);
        let next_result_id = AtomicUsize::new(0);

        // Initialize statistics
        let stats = Arc::new(std::sync::Mutex::new(LoaderStats {
            batches_processed: 0,
            total_processing_time: Duration::from_secs(0),
            average_batch_time: Duration::from_secs(0),
            peak_memory_usage: None,
            work_stealing_events: 0,
        }));

        // Initialize NUMA scheduler if enabled
        let (numa_scheduler, numa_assignments) = if let Some(numa_config) = &config.numa_config {
            if numa_config.enabled {
                match NumaScheduler::new(numa_config.clone()) {
                    Ok(mut scheduler) => {
                        match scheduler.assign_workers(config.num_workers) {
                            Ok(assignments) => (Some(scheduler), assignments),
                            Err(_) => {
                                // Fall back to no NUMA scheduling if assignment fails
                                (None, Vec::new())
                            }
                        }
                    }
                    Err(_) => {
                        // Fall back to no NUMA scheduling if creation fails
                        (None, Vec::new())
                    }
                }
            } else {
                (None, Vec::new())
            }
        } else {
            (None, Vec::new())
        };

        // Generate work tasks
        let mut task_id = 0;
        for batch_start in (0..indices.len()).step_by(config.batch_size) {
            let batch_end = if config.drop_last {
                let end = batch_start + config.batch_size;
                if end <= indices.len() {
                    end
                } else {
                    continue;
                }
            } else {
                (batch_start + config.batch_size).min(indices.len())
            };

            if batch_start >= indices.len()
                || (config.drop_last && batch_end - batch_start < config.batch_size)
            {
                break;
            }

            let batch_indices = indices[batch_start..batch_end].to_vec();
            let task = WorkTask {
                batch_indices,
                task_id,
            };

            work_queue.push(task);
            task_id += 1;
        }

        // Start worker threads
        let mut worker_handles = Vec::new();

        for worker_id in 0..config.num_workers {
            let dataset_clone = Arc::clone(&dataset);
            let config_clone = config.clone();
            let work_queue_clone = Arc::clone(&work_queue);
            let results_clone = Arc::clone(&results);
            let shutdown_clone = Arc::clone(&shutdown_signal);
            let stats_clone = Arc::clone(&stats);

            // Get NUMA assignment for this worker if available
            let numa_assignment = numa_assignments
                .iter()
                .find(|assignment| assignment.worker_id == worker_id)
                .cloned();

            let handle = thread::spawn(move || {
                // Set CPU affinity if NUMA assignment is available
                if let Some(assignment) = &numa_assignment {
                    let _ = NumaScheduler::set_thread_affinity(assignment);
                }

                Self::worker_thread(
                    worker_id,
                    dataset_clone,
                    config_clone,
                    work_queue_clone,
                    results_clone,
                    shutdown_clone,
                    stats_clone,
                    numa_assignment,
                )
            });

            worker_handles.push(handle);
        }

        Ok(Self {
            dataset,
            config,
            work_queue,
            results,
            worker_handles,
            shutdown_signal,
            next_task_id,
            next_result_id,
            total_tasks,
            stats,
            numa_scheduler,
            numa_assignments,
        })
    }

    /// Worker thread function
    #[allow(clippy::too_many_arguments)]
    fn worker_thread(
        worker_id: usize,
        dataset: Arc<D>,
        config: DataLoaderConfig,
        work_queue: Arc<WorkStealingQueue<WorkTask>>,
        results: Arc<std::sync::Mutex<std::collections::BTreeMap<usize, WorkResult<T>>>>,
        shutdown_signal: Arc<AtomicBool>,
        _stats: Arc<std::sync::Mutex<LoaderStats>>,
        _numa_assignment: Option<NumaWorkerAssignment>,
    ) -> WorkerStats {
        let mut worker_stats = WorkerStats {
            worker_id,
            tasks_processed: 0,
            tasks_stolen: 0,
            total_processing_time: Duration::from_secs(0),
            cache_hits: 0,
        };

        while !shutdown_signal.load(Ordering::Relaxed) {
            // Try to get work (either from own queue or steal from others)
            if let Some(task) = work_queue.wait_for_work(worker_id, Some(100)) {
                let start_time = Instant::now();

                // Process the task
                let batch_result = Self::process_task(&dataset, &task, &config);
                let processing_time = start_time.elapsed();

                // Store the result
                let work_result = WorkResult {
                    task_id: task.task_id,
                    batch_result,
                    processing_time,
                };

                {
                    let mut results_map = results.lock().unwrap();
                    results_map.insert(task.task_id, work_result);
                }

                // Update worker statistics
                worker_stats.tasks_processed += 1;
                worker_stats.total_processing_time += processing_time;

                // Check if this task was stolen (simple heuristic)
                if worker_id != task.task_id % config.num_workers {
                    worker_stats.tasks_stolen += 1;
                }
            } else if work_queue.is_empty() {
                // No more work available
                break;
            }
        }

        worker_stats
    }

    /// Process a single work task
    fn process_task(
        dataset: &Arc<D>,
        task: &WorkTask,
        config: &DataLoaderConfig,
    ) -> Result<BatchResult<T>> {
        let mut batch = Vec::with_capacity(task.batch_indices.len());

        for &idx in &task.batch_indices {
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

        if config.collate_batches {
            // Collate the batch into stacked tensors
            let collate_fn = DefaultCollate;
            let (features, labels) = collate_fn.collate(batch)?;
            Ok(BatchResult::Collated(features, labels))
        } else {
            // Return individual samples
            Ok(BatchResult::Samples(batch))
        }
    }

    /// Get the next processed batch (blocking)
    pub fn next_batch(&self) -> Option<Result<BatchResult<T>>> {
        let current_id = self.next_result_id.fetch_add(1, Ordering::Relaxed);

        if current_id >= self.total_tasks {
            return None;
        }

        // Wait for the result to be available with timeout
        let start_time = Instant::now();
        let timeout = Duration::from_secs(10); // 10 second timeout

        loop {
            {
                let mut results_map = self.results.lock().unwrap();
                if let Some(result) = results_map.remove(&current_id) {
                    // Update statistics
                    {
                        let mut stats = self.stats.lock().unwrap();
                        stats.batches_processed += 1;
                        stats.total_processing_time += result.processing_time;
                        stats.average_batch_time =
                            stats.total_processing_time / stats.batches_processed as u32;
                    }

                    return Some(result.batch_result);
                }
            }

            // Check timeout
            if start_time.elapsed() > timeout {
                return Some(Err(TensorError::invalid_argument(format!(
                    "Timeout waiting for batch {current_id}"
                ))));
            }

            // Check if work is done (more robust condition)
            // Only return None if:
            // 1. Work queue is empty
            // 2. All worker threads have finished (not just current_id check)
            // 3. We've waited a reasonable amount of time
            if self.work_queue.is_empty() && start_time.elapsed() > Duration::from_millis(100) {
                // Check if any workers are still alive
                let all_workers_done = self
                    .worker_handles
                    .iter()
                    .all(|handle| handle.is_finished());
                if all_workers_done {
                    return None;
                }
            }

            // Brief sleep to avoid busy waiting
            thread::sleep(Duration::from_millis(1));
        }
    }

    /// Get loader statistics
    pub fn get_stats(&self) -> LoaderStats {
        self.stats.lock().unwrap().clone()
    }

    /// Get work queue statistics
    pub fn get_queue_stats(&self) -> (Vec<usize>, usize, bool) {
        (
            self.work_queue.queue_lengths(),
            self.total_tasks, // Use planned total batches instead of current queue count
            self.work_queue.is_empty(),
        )
    }

    /// Get NUMA assignment statistics if NUMA scheduling is enabled
    pub fn get_numa_stats(&self) -> Option<crate::numa_scheduler::NumaAssignmentStats> {
        self.numa_scheduler
            .as_ref()
            .map(|scheduler| scheduler.get_assignment_stats())
    }

    /// Get NUMA topology information if NUMA scheduling is enabled
    pub fn get_numa_topology(&self) -> Option<&crate::numa_scheduler::NumaTopology> {
        self.numa_scheduler
            .as_ref()
            .map(|scheduler| scheduler.topology())
    }

    /// Shutdown the loader and collect worker statistics
    pub fn shutdown(self) -> Result<Vec<WorkerStats>> {
        // Signal shutdown
        self.shutdown_signal.store(true, Ordering::Relaxed);
        self.work_queue.shutdown();

        // Collect worker statistics
        let mut worker_stats = Vec::new();
        for handle in self.worker_handles {
            match handle.join() {
                Ok(stats) => worker_stats.push(stats),
                Err(_) => {
                    return Err(TensorError::invalid_argument(
                        "Worker thread panicked".to_string(),
                    ))
                }
            }
        }

        Ok(worker_stats)
    }
}

impl<T, D: Dataset<T> + Send + Sync + 'static> Iterator for EnhancedDataLoader<T, D>
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
        self.next_batch()
    }
}

/// Builder for EnhancedDataLoader with convenient configuration
pub struct EnhancedDataLoaderBuilder {
    config: DataLoaderConfig,
}

impl EnhancedDataLoaderBuilder {
    /// Create a new builder with default configuration
    pub fn new() -> Self {
        Self {
            config: DataLoaderConfig::default(),
        }
    }

    /// Set the batch size
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.config.batch_size = batch_size;
        self
    }

    /// Set the number of worker threads
    pub fn num_workers(mut self, num_workers: usize) -> Self {
        self.config.num_workers = num_workers;
        self
    }

    /// Enable or disable dropping the last incomplete batch
    pub fn drop_last(mut self, drop_last: bool) -> Self {
        self.config.drop_last = drop_last;
        self
    }

    /// Enable automatic batch collation
    pub fn collate_batches(mut self, collate: bool) -> Self {
        self.config.collate_batches = collate;
        self
    }

    /// Set the prefetch factor for background data loading
    pub fn prefetch_factor(mut self, prefetch_factor: usize) -> Self {
        self.config.prefetch_factor = prefetch_factor;
        self
    }

    /// Set target device for tensor loading
    pub fn target_device(mut self, device: Device) -> Self {
        self.config.target_device = Some(device);
        self
    }

    /// Enable NUMA-aware scheduling with default configuration
    pub fn numa_scheduling(mut self) -> Self {
        self.config.numa_config = Some(crate::numa_scheduler::NumaConfig::default());
        self
    }

    /// Set custom NUMA configuration
    pub fn numa_config(mut self, numa_config: crate::numa_scheduler::NumaConfig) -> Self {
        self.config.numa_config = Some(numa_config);
        self
    }

    /// Build the enhanced DataLoader
    pub fn build<T, D: Dataset<T> + Send + Sync + 'static, S: Sampler>(
        self,
        dataset: D,
        sampler: S,
    ) -> Result<EnhancedDataLoader<T, D>>
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
        EnhancedDataLoader::new(dataset, self.config, sampler)
    }
}

impl Default for EnhancedDataLoaderBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{SequentialSampler, TensorDataset};
    use tenflowers_core::Tensor;

    #[test]
    fn test_enhanced_dataloader_creation() {
        let features =
            Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[4, 2]).unwrap();
        let labels = Tensor::<f32>::from_vec(vec![0.0, 1.0, 0.0, 1.0], &[4]).unwrap();

        let dataset = TensorDataset::new(features, labels);
        let sampler = SequentialSampler::new();

        let loader = EnhancedDataLoaderBuilder::new()
            .batch_size(2)
            .num_workers(2)
            .build(dataset, sampler)
            .unwrap();

        // Check queue statistics
        let (queue_lengths, total_tasks, _is_empty) = loader.get_queue_stats();
        assert_eq!(queue_lengths.len(), 2); // 2 workers
        assert_eq!(total_tasks, 2); // 4 samples / 2 batch_size = 2 batches
                                    // Note: don't check is_empty since worker threads may consume tasks immediately
    }

    #[test]
    fn test_enhanced_dataloader_processing() {
        let features =
            Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();
        let labels = Tensor::<f32>::from_vec(vec![0.0, 1.0, 2.0], &[3]).unwrap();

        let dataset = TensorDataset::new(features, labels);
        let sampler = SequentialSampler::new();

        let mut loader = EnhancedDataLoaderBuilder::new()
            .batch_size(2)
            .num_workers(1)
            .collate_batches(true)
            .build(dataset, sampler)
            .unwrap();

        // Get first batch
        let batch1 = loader.next().unwrap().unwrap();
        match batch1 {
            BatchResult::Collated(features, labels) => {
                assert_eq!(features.shape().dims(), &[2, 2]); // batch_size=2, feature_size=2
                assert_eq!(labels.shape().dims(), &[2]); // batch_size=2
            }
            _ => assert!(false, "Expected collated batch"),
        }

        // Get second batch (partial)
        let batch2 = loader.next().unwrap().unwrap();
        match batch2 {
            BatchResult::Collated(features, labels) => {
                assert_eq!(features.shape().dims(), &[1, 2]); // batch_size=1, feature_size=2
                assert_eq!(labels.shape().dims(), &[1]); // batch_size=1
            }
            _ => assert!(false, "Expected collated batch"),
        }

        // No more batches
        assert!(loader.next().is_none());

        // Check statistics
        let stats = loader.get_stats();
        assert_eq!(stats.batches_processed, 2);
        assert!(stats.average_batch_time > Duration::from_secs(0));
    }

    #[test]
    fn test_enhanced_dataloader_worker_stats() {
        let features =
            Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[4, 2]).unwrap();
        let labels = Tensor::<f32>::from_vec(vec![0.0, 1.0, 0.0, 1.0], &[4]).unwrap();

        let dataset = TensorDataset::new(features, labels);
        let sampler = SequentialSampler::new();

        let loader = EnhancedDataLoaderBuilder::new()
            .batch_size(1)
            .num_workers(2)
            .build(dataset, sampler)
            .unwrap();

        // Process all batches
        let _batches: Vec<_> = loader.collect();

        // Shutdown and get worker statistics
        // Note: In real usage, we'd need to handle the loader properly
        // For testing, we just verify the structure exists
    }
}
