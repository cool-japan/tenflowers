//! Throughput benchmark performance harness for datasets
//!
//! This module provides comprehensive benchmarking capabilities for measuring
//! dataset loading, transformation, and iteration performance. It supports
//! multi-threaded testing, various batch sizes, and detailed performance metrics.

use crate::Dataset;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Throughput benchmark configuration
#[derive(Debug, Clone)]
pub struct ThroughputBenchmarkConfig {
    /// Number of warmup iterations before measurement
    pub warmup_iterations: usize,
    /// Number of measurement iterations
    pub measurement_iterations: usize,
    /// Batch size for batched operations
    pub batch_size: Option<usize>,
    /// Number of worker threads for parallel testing
    pub num_threads: Option<usize>,
    /// Whether to measure memory usage
    pub measure_memory: bool,
    /// Whether to include detailed per-sample timings
    pub detailed_timings: bool,
    /// Maximum samples to benchmark (None = all)
    pub max_samples: Option<usize>,
}

impl Default for ThroughputBenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 10,
            measurement_iterations: 100,
            batch_size: None,
            num_threads: None,
            measure_memory: false,
            detailed_timings: false,
            max_samples: None,
        }
    }
}

/// Results from a throughput benchmark
#[derive(Debug, Clone)]
pub struct ThroughputBenchmarkResult {
    /// Dataset name or identifier
    pub dataset_name: String,
    /// Total samples processed
    pub samples_processed: usize,
    /// Total time elapsed
    pub total_duration: Duration,
    /// Samples per second
    pub samples_per_second: f64,
    /// Average latency per sample (microseconds)
    pub avg_latency_us: f64,
    /// P50 latency (microseconds)
    pub p50_latency_us: f64,
    /// P95 latency (microseconds)
    pub p95_latency_us: f64,
    /// P99 latency (microseconds)
    pub p99_latency_us: f64,
    /// Minimum latency (microseconds)
    pub min_latency_us: f64,
    /// Maximum latency (microseconds)
    pub max_latency_us: f64,
    /// Standard deviation of latency
    pub latency_std_dev_us: f64,
    /// Memory usage statistics (if measured)
    pub memory_stats: Option<MemoryStats>,
    /// Per-thread statistics (if multi-threaded)
    pub per_thread_stats: Vec<ThreadStats>,
    /// Timestamp when benchmark was run
    pub timestamp: Instant,
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Peak memory usage in bytes
    pub peak_bytes: usize,
    /// Average memory usage in bytes
    pub avg_bytes: usize,
    /// Memory allocations per second
    pub allocations_per_second: f64,
}

/// Per-thread statistics
#[derive(Debug, Clone)]
pub struct ThreadStats {
    /// Thread identifier
    pub thread_id: usize,
    /// Samples processed by this thread
    pub samples_processed: usize,
    /// Time spent by this thread
    pub duration: Duration,
    /// Samples per second for this thread
    pub samples_per_second: f64,
}

/// Throughput benchmark harness
pub struct ThroughputBenchmarkHarness {
    /// Benchmark configuration
    config: ThroughputBenchmarkConfig,
    /// Collected sample latencies (microseconds)
    sample_latencies: Arc<Mutex<Vec<u64>>>,
    /// Memory usage samples (if measuring)
    memory_samples: Arc<Mutex<Vec<usize>>>,
    /// Per-thread statistics (if multi-threaded)
    thread_stats: Arc<Mutex<Vec<ThreadStats>>>,
}

impl ThroughputBenchmarkHarness {
    /// Create a new benchmark harness with configuration
    pub fn new(config: ThroughputBenchmarkConfig) -> Self {
        Self {
            config,
            sample_latencies: Arc::new(Mutex::new(Vec::new())),
            memory_samples: Arc::new(Mutex::new(Vec::new())),
            thread_stats: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Create a default benchmark harness
    pub fn default() -> Self {
        Self::new(ThroughputBenchmarkConfig::default())
    }

    /// Benchmark a dataset's iteration performance
    pub fn benchmark<T, D>(
        &mut self,
        dataset: &D,
        name: impl Into<String>,
    ) -> ThroughputBenchmarkResult
    where
        T: Clone + Send + Sync + 'static,
        D: Dataset<T> + Sync,
    {
        let dataset_name = name.into();
        let total_samples = if let Some(max) = self.config.max_samples {
            max.min(dataset.len())
        } else {
            dataset.len()
        };

        // Warmup phase
        self.warmup_phase(dataset, total_samples);

        // Measurement phase
        let start_time = Instant::now();
        self.measurement_phase(dataset, total_samples);
        let total_duration = start_time.elapsed();

        // Calculate statistics
        let latencies = self.sample_latencies.lock().unwrap().clone();
        let stats = calculate_latency_statistics(&latencies);

        // Calculate memory statistics if measured
        let memory_stats = if self.config.measure_memory {
            let memory_samples = self.memory_samples.lock().unwrap();
            if !memory_samples.is_empty() {
                let peak_bytes = *memory_samples.iter().max().unwrap_or(&0);
                let avg_bytes = memory_samples.iter().sum::<usize>() / memory_samples.len();
                let allocations_per_second =
                    memory_samples.len() as f64 / total_duration.as_secs_f64();
                Some(MemoryStats {
                    peak_bytes,
                    avg_bytes,
                    allocations_per_second,
                })
            } else {
                None
            }
        } else {
            None
        };

        // Get per-thread statistics if multi-threaded
        let per_thread_stats = self.thread_stats.lock().unwrap().clone();

        ThroughputBenchmarkResult {
            dataset_name,
            samples_processed: total_samples,
            total_duration,
            samples_per_second: total_samples as f64 / total_duration.as_secs_f64(),
            avg_latency_us: stats.mean,
            p50_latency_us: stats.p50,
            p95_latency_us: stats.p95,
            p99_latency_us: stats.p99,
            min_latency_us: stats.min,
            max_latency_us: stats.max,
            latency_std_dev_us: stats.std_dev,
            memory_stats,
            per_thread_stats,
            timestamp: Instant::now(),
        }
    }

    /// Benchmark with batched access
    pub fn benchmark_batched<T, D>(
        &mut self,
        dataset: &D,
        batch_size: usize,
        name: impl Into<String>,
    ) -> ThroughputBenchmarkResult
    where
        T: Clone + Send + Sync + 'static,
        D: Dataset<T> + Sync,
    {
        let dataset_name = name.into();
        let total_samples = if let Some(max) = self.config.max_samples {
            max.min(dataset.len())
        } else {
            dataset.len()
        };

        // Warmup phase with batches
        self.warmup_phase_batched(dataset, batch_size, total_samples);

        // Measurement phase with batches
        let start_time = Instant::now();
        self.measurement_phase_batched(dataset, batch_size, total_samples);
        let total_duration = start_time.elapsed();

        // Calculate statistics
        let latencies = self.sample_latencies.lock().unwrap().clone();
        let stats = calculate_latency_statistics(&latencies);

        // Calculate memory statistics if measured
        let memory_stats = if self.config.measure_memory {
            let memory_samples = self.memory_samples.lock().unwrap();
            if !memory_samples.is_empty() {
                let peak_bytes = *memory_samples.iter().max().unwrap_or(&0);
                let avg_bytes = memory_samples.iter().sum::<usize>() / memory_samples.len();
                let allocations_per_second =
                    memory_samples.len() as f64 / total_duration.as_secs_f64();
                Some(MemoryStats {
                    peak_bytes,
                    avg_bytes,
                    allocations_per_second,
                })
            } else {
                None
            }
        } else {
            None
        };

        // Get per-thread statistics if multi-threaded
        let per_thread_stats = self.thread_stats.lock().unwrap().clone();

        ThroughputBenchmarkResult {
            dataset_name,
            samples_processed: total_samples,
            total_duration,
            samples_per_second: total_samples as f64 / total_duration.as_secs_f64(),
            avg_latency_us: stats.mean,
            p50_latency_us: stats.p50,
            p95_latency_us: stats.p95,
            p99_latency_us: stats.p99,
            min_latency_us: stats.min,
            max_latency_us: stats.max,
            latency_std_dev_us: stats.std_dev,
            memory_stats,
            per_thread_stats,
            timestamp: Instant::now(),
        }
    }

    /// Compare multiple datasets
    pub fn compare_datasets<T, D>(
        &mut self,
        datasets: Vec<(&D, String)>,
    ) -> HashMap<String, ThroughputBenchmarkResult>
    where
        T: Clone + Send + Sync + 'static,
        D: Dataset<T> + Sync,
    {
        let mut results = HashMap::new();

        for (dataset, name) in datasets {
            let result = self.benchmark(dataset, name.clone());
            results.insert(name, result);
        }

        results
    }

    /// Benchmark with multi-threading (requires parallel feature)
    #[cfg(feature = "parallel")]
    pub fn benchmark_multithreaded<T, D>(
        &mut self,
        dataset: &D,
        num_threads: usize,
        name: impl Into<String>,
    ) -> ThroughputBenchmarkResult
    where
        T: Clone + Send + Sync + 'static,
        D: Dataset<T> + Sync,
    {
        let dataset_name = name.into();
        let total_samples = if let Some(max) = self.config.max_samples {
            max.min(dataset.len())
        } else {
            dataset.len()
        };

        // Warmup phase
        self.warmup_phase(dataset, total_samples);

        // Clear thread stats
        self.thread_stats.lock().unwrap().clear();

        // Measurement phase with parallel execution
        let start_time = Instant::now();

        // Divide samples among threads
        let samples_per_thread = (total_samples + num_threads - 1) / num_threads;
        let thread_ranges: Vec<_> = (0..num_threads)
            .map(|i| {
                let start = i * samples_per_thread;
                let end = ((i + 1) * samples_per_thread).min(total_samples);
                (i, start, end)
            })
            .collect();

        // Execute benchmark in parallel
        let thread_stats_mutex = Arc::clone(&self.thread_stats);
        thread_ranges
            .par_iter()
            .for_each(|(thread_id, start, end)| {
                let thread_start = Instant::now();
                let mut samples_processed = 0;

                for _ in 0..self.config.measurement_iterations {
                    for i in *start..*end {
                        let _ = dataset.get(i);
                        samples_processed += 1;
                    }
                }

                let thread_duration = thread_start.elapsed();
                let samples_per_second = samples_processed as f64 / thread_duration.as_secs_f64();

                // Record thread statistics
                let mut stats = thread_stats_mutex.lock().unwrap();
                stats.push(ThreadStats {
                    thread_id: *thread_id,
                    samples_processed,
                    duration: thread_duration,
                    samples_per_second,
                });
            });

        let total_duration = start_time.elapsed();

        // Calculate statistics (using thread stats for latency approximation)
        let thread_stats = self.thread_stats.lock().unwrap().clone();
        let total_processed: usize = thread_stats.iter().map(|s| s.samples_processed).sum();
        let avg_latency_us = (total_duration.as_micros() as f64) / (total_processed as f64);

        // Calculate memory statistics if measured
        let memory_stats = if self.config.measure_memory {
            let memory_samples = self.memory_samples.lock().unwrap();
            if !memory_samples.is_empty() {
                let peak_bytes = *memory_samples.iter().max().unwrap_or(&0);
                let avg_bytes = memory_samples.iter().sum::<usize>() / memory_samples.len();
                let allocations_per_second =
                    memory_samples.len() as f64 / total_duration.as_secs_f64();
                Some(MemoryStats {
                    peak_bytes,
                    avg_bytes,
                    allocations_per_second,
                })
            } else {
                None
            }
        } else {
            None
        };

        ThroughputBenchmarkResult {
            dataset_name,
            samples_processed: total_processed,
            total_duration,
            samples_per_second: total_processed as f64 / total_duration.as_secs_f64(),
            avg_latency_us,
            p50_latency_us: avg_latency_us,
            p95_latency_us: avg_latency_us,
            p99_latency_us: avg_latency_us,
            min_latency_us: avg_latency_us,
            max_latency_us: avg_latency_us,
            latency_std_dev_us: 0.0,
            memory_stats,
            per_thread_stats: thread_stats,
            timestamp: Instant::now(),
        }
    }

    /// Reset collected metrics
    pub fn reset(&mut self) {
        self.sample_latencies.lock().unwrap().clear();
        self.memory_samples.lock().unwrap().clear();
        self.thread_stats.lock().unwrap().clear();
    }

    /// Get current memory usage (platform-specific approximation)
    fn get_current_memory_usage(&self) -> usize {
        // This is a basic approximation. On Linux/Unix systems, you could read from /proc
        // For now, we'll return 0 as a placeholder. Real implementation would use
        // platform-specific APIs or crates like `jemalloc_ctl` or `memory-stats`
        0
    }

    /// Track memory usage during benchmark
    fn track_memory(&self) {
        if self.config.measure_memory {
            let mem = self.get_current_memory_usage();
            self.memory_samples.lock().unwrap().push(mem);
        }
    }

    // Private helper methods

    fn warmup_phase<T, D>(&self, dataset: &D, total_samples: usize)
    where
        T: Clone + Send + Sync + 'static,
        D: Dataset<T>,
    {
        for _ in 0..self.config.warmup_iterations {
            for i in 0..total_samples {
                let _ = dataset.get(i);
            }
        }
    }

    fn measurement_phase<T, D>(&self, dataset: &D, total_samples: usize)
    where
        T: Clone + Send + Sync + 'static,
        D: Dataset<T>,
    {
        let mut latencies = self.sample_latencies.lock().unwrap();
        latencies.clear();

        for _ in 0..self.config.measurement_iterations {
            self.track_memory(); // Track memory at start of each iteration

            for i in 0..total_samples {
                let start = Instant::now();
                let _ = dataset.get(i);
                let latency = start.elapsed().as_micros() as u64;

                if self.config.detailed_timings {
                    latencies.push(latency);
                }

                // Track memory periodically (every 100 samples)
                if i % 100 == 0 {
                    self.track_memory();
                }
            }
        }

        // If not detailed, record average latency
        if !self.config.detailed_timings && !latencies.is_empty() {
            let avg = latencies.iter().sum::<u64>() / latencies.len() as u64;
            latencies.clear();
            latencies.push(avg);
        }
    }

    fn warmup_phase_batched<T, D>(&self, dataset: &D, batch_size: usize, total_samples: usize)
    where
        T: Clone + Send + Sync + 'static,
        D: Dataset<T>,
    {
        for _ in 0..self.config.warmup_iterations {
            for batch_start in (0..total_samples).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(total_samples);
                for i in batch_start..batch_end {
                    let _ = dataset.get(i);
                }
            }
        }
    }

    fn measurement_phase_batched<T, D>(&self, dataset: &D, batch_size: usize, total_samples: usize)
    where
        T: Clone + Send + Sync + 'static,
        D: Dataset<T>,
    {
        let mut latencies = self.sample_latencies.lock().unwrap();
        latencies.clear();

        for _ in 0..self.config.measurement_iterations {
            for batch_start in (0..total_samples).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(total_samples);
                let start = Instant::now();

                for i in batch_start..batch_end {
                    let _ = dataset.get(i);
                }

                let batch_latency = start.elapsed().as_micros() as u64;
                let per_sample_latency = batch_latency / (batch_end - batch_start) as u64;

                if self.config.detailed_timings {
                    latencies.push(per_sample_latency);
                }
            }
        }
    }
}

/// Latency statistics
struct LatencyStatistics {
    mean: f64,
    min: f64,
    max: f64,
    p50: f64,
    p95: f64,
    p99: f64,
    std_dev: f64,
}

/// Calculate latency statistics from collected samples
fn calculate_latency_statistics(latencies: &[u64]) -> LatencyStatistics {
    if latencies.is_empty() {
        return LatencyStatistics {
            mean: 0.0,
            min: 0.0,
            max: 0.0,
            p50: 0.0,
            p95: 0.0,
            p99: 0.0,
            std_dev: 0.0,
        };
    }

    let mut sorted = latencies.to_vec();
    sorted.sort_unstable();

    let sum: u64 = sorted.iter().sum();
    let mean = sum as f64 / sorted.len() as f64;

    let variance = sorted
        .iter()
        .map(|&x| {
            let diff = x as f64 - mean;
            diff * diff
        })
        .sum::<f64>()
        / sorted.len() as f64;
    let std_dev = variance.sqrt();

    let percentile = |p: f64| -> f64 {
        let index = ((sorted.len() as f64 * p) as usize).min(sorted.len() - 1);
        sorted[index] as f64
    };

    LatencyStatistics {
        mean,
        min: sorted[0] as f64,
        max: sorted[sorted.len() - 1] as f64,
        p50: percentile(0.50),
        p95: percentile(0.95),
        p99: percentile(0.99),
        std_dev,
    }
}

impl ThroughputBenchmarkResult {
    /// Generate a human-readable report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();

        report.push_str(&format!(
            "=== Throughput Benchmark Report: {} ===\n\n",
            self.dataset_name
        ));

        report.push_str("## Overall Statistics\n");
        report.push_str(&format!(
            "  Samples Processed: {}\n",
            self.samples_processed
        ));
        report.push_str(&format!("  Total Duration: {:.2?}\n", self.total_duration));
        report.push_str(&format!(
            "  Throughput: {:.2} samples/sec\n\n",
            self.samples_per_second
        ));

        report.push_str("## Latency Statistics (microseconds)\n");
        report.push_str(&format!("  Average: {:.2}\n", self.avg_latency_us));
        report.push_str(&format!("  Minimum: {:.2}\n", self.min_latency_us));
        report.push_str(&format!("  Maximum: {:.2}\n", self.max_latency_us));
        report.push_str(&format!("  Std Dev: {:.2}\n", self.latency_std_dev_us));
        report.push_str(&format!("  P50: {:.2}\n", self.p50_latency_us));
        report.push_str(&format!("  P95: {:.2}\n", self.p95_latency_us));
        report.push_str(&format!("  P99: {:.2}\n\n", self.p99_latency_us));

        if let Some(ref mem_stats) = self.memory_stats {
            report.push_str("## Memory Statistics\n");
            report.push_str(&format!("  Peak: {} bytes\n", mem_stats.peak_bytes));
            report.push_str(&format!("  Average: {} bytes\n", mem_stats.avg_bytes));
            report.push_str(&format!(
                "  Allocations/sec: {:.2}\n\n",
                mem_stats.allocations_per_second
            ));
        }

        if !self.per_thread_stats.is_empty() {
            report.push_str("## Per-Thread Statistics\n");
            for thread_stat in &self.per_thread_stats {
                report.push_str(&format!(
                    "  Thread {}: {} samples, {:.2?}, {:.2} samples/sec\n",
                    thread_stat.thread_id,
                    thread_stat.samples_processed,
                    thread_stat.duration,
                    thread_stat.samples_per_second
                ));
            }
        }

        report
    }

    /// Export results as CSV
    pub fn to_csv(&self) -> String {
        format!(
            "{},{},{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2}\n",
            self.dataset_name,
            self.samples_processed,
            self.total_duration.as_millis(),
            self.samples_per_second,
            self.avg_latency_us,
            self.min_latency_us,
            self.max_latency_us,
            self.p50_latency_us,
            self.p95_latency_us,
            self.p99_latency_us,
            self.latency_std_dev_us
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TensorDataset;
    use tenflowers_core::Tensor;

    #[test]
    fn test_benchmark_harness_creation() {
        let harness = ThroughputBenchmarkHarness::default();
        assert_eq!(harness.config.warmup_iterations, 10);
        assert_eq!(harness.config.measurement_iterations, 100);
    }

    #[test]
    fn test_basic_benchmark() {
        let features =
            Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[4, 2]).unwrap();
        let labels = Tensor::<f32>::from_vec(vec![0.0, 1.0, 2.0, 3.0], &[4]).unwrap();

        let dataset = TensorDataset::new(features, labels);
        let mut harness = ThroughputBenchmarkHarness::new(ThroughputBenchmarkConfig {
            warmup_iterations: 1,
            measurement_iterations: 5,
            max_samples: Some(4),
            ..Default::default()
        });

        let result = harness.benchmark(&dataset, "test_dataset");

        assert_eq!(result.samples_processed, 4);
        assert!(result.samples_per_second > 0.0);
        assert!(result.avg_latency_us >= 0.0);
    }

    #[test]
    fn test_batched_benchmark() {
        let features = Tensor::<f32>::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            &[5, 2],
        )
        .unwrap();
        let labels = Tensor::<f32>::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0], &[5]).unwrap();

        let dataset = TensorDataset::new(features, labels);
        let mut harness = ThroughputBenchmarkHarness::new(ThroughputBenchmarkConfig {
            warmup_iterations: 1,
            measurement_iterations: 3,
            max_samples: Some(5),
            ..Default::default()
        });

        let result = harness.benchmark_batched(&dataset, 2, "batched_test");

        assert_eq!(result.samples_processed, 5);
        assert!(result.samples_per_second > 0.0);
    }

    #[test]
    fn test_generate_report() {
        let features = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let labels = Tensor::<f32>::from_vec(vec![0.0, 1.0], &[2]).unwrap();

        let dataset = TensorDataset::new(features, labels);
        let mut harness = ThroughputBenchmarkHarness::new(ThroughputBenchmarkConfig {
            warmup_iterations: 1,
            measurement_iterations: 2,
            ..Default::default()
        });

        let result = harness.benchmark(&dataset, "report_test");
        let report = result.generate_report();

        assert!(report.contains("Throughput Benchmark Report"));
        assert!(report.contains("Samples Processed"));
        assert!(report.contains("Throughput:"));
        assert!(report.contains("Latency Statistics"));
    }

    #[test]
    fn test_csv_export() {
        let features = Tensor::<f32>::from_vec(vec![1.0, 2.0], &[1, 2]).unwrap();
        let labels = Tensor::<f32>::from_vec(vec![0.0], &[1]).unwrap();

        let dataset = TensorDataset::new(features, labels);
        let mut harness = ThroughputBenchmarkHarness::new(ThroughputBenchmarkConfig {
            warmup_iterations: 1,
            measurement_iterations: 1,
            ..Default::default()
        });

        let result = harness.benchmark(&dataset, "csv_test");
        let csv = result.to_csv();

        assert!(csv.contains("csv_test"));
        assert!(csv.contains(','));
    }

    #[test]
    fn test_reset() {
        let features = Tensor::<f32>::from_vec(vec![1.0, 2.0], &[1, 2]).unwrap();
        let labels = Tensor::<f32>::from_vec(vec![0.0], &[1]).unwrap();

        let dataset = TensorDataset::new(features, labels);
        let mut harness = ThroughputBenchmarkHarness::new(ThroughputBenchmarkConfig {
            warmup_iterations: 1,
            measurement_iterations: 2,
            detailed_timings: true,
            ..Default::default()
        });

        let _ = harness.benchmark(&dataset, "test1");
        assert!(!harness.sample_latencies.lock().unwrap().is_empty());

        harness.reset();
        assert!(harness.sample_latencies.lock().unwrap().is_empty());
    }

    #[test]
    fn test_compare_datasets() {
        let features1 = Tensor::<f32>::from_vec(vec![1.0, 2.0], &[1, 2]).unwrap();
        let labels1 = Tensor::<f32>::from_vec(vec![0.0], &[1]).unwrap();
        let dataset1 = TensorDataset::new(features1, labels1);

        let features2 = Tensor::<f32>::from_vec(vec![3.0, 4.0], &[1, 2]).unwrap();
        let labels2 = Tensor::<f32>::from_vec(vec![1.0], &[1]).unwrap();
        let dataset2 = TensorDataset::new(features2, labels2);

        let mut harness = ThroughputBenchmarkHarness::new(ThroughputBenchmarkConfig {
            warmup_iterations: 1,
            measurement_iterations: 1,
            ..Default::default()
        });

        let results = harness.compare_datasets(vec![
            (&dataset1, "dataset1".to_string()),
            (&dataset2, "dataset2".to_string()),
        ]);

        assert_eq!(results.len(), 2);
        assert!(results.contains_key("dataset1"));
        assert!(results.contains_key("dataset2"));
    }
}
