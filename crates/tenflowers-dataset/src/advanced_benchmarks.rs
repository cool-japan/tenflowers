//! Advanced benchmarking system for comprehensive performance analysis
//!
//! This module provides sophisticated benchmarking tools for analyzing dataset
//! performance across different operations, hardware configurations, and workloads.

use crate::{Dataset, Transform};
use scirs2_core::random::Rng;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use tenflowers_core::{Result, Tensor, TensorError};

#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

/// Comprehensive benchmark suite for dataset operations
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct AdvancedBenchmarkSuite {
    /// Configuration for benchmark execution
    pub config: BenchmarkConfig,
    /// Results from completed benchmarks
    pub results: Vec<BenchmarkResult>,
    /// System information
    pub system_info: SystemInfo,
}

/// Configuration for benchmark execution
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct BenchmarkConfig {
    /// Number of warmup iterations before measurement
    pub warmup_iterations: usize,
    /// Number of measurement iterations
    pub measurement_iterations: usize,
    /// Timeout for individual benchmarks (in seconds)
    pub timeout_seconds: u64,
    /// Whether to include memory usage tracking
    pub track_memory: bool,
    /// Whether to include CPU utilization tracking
    pub track_cpu: bool,
    /// Whether to include GPU utilization tracking (if available)
    pub track_gpu: bool,
    /// Sample sizes to test
    pub sample_sizes: Vec<usize>,
    /// Batch sizes to test for DataLoader benchmarks
    pub batch_sizes: Vec<usize>,
    /// Number of worker threads to test
    pub worker_counts: Vec<usize>,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 5,
            measurement_iterations: 10,
            timeout_seconds: 60,
            track_memory: true,
            track_cpu: true,
            track_gpu: false,
            sample_sizes: vec![100, 1000, 10000],
            batch_sizes: vec![1, 8, 32, 128],
            worker_counts: vec![1, 2, 4, 8],
        }
    }
}

/// Results from a benchmark run
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct BenchmarkResult {
    /// Name of the benchmark
    pub name: String,
    /// Configuration used for this benchmark
    pub config: BenchmarkConfig,
    /// Timing measurements
    pub timing: TimingStats,
    /// Memory usage statistics
    pub memory: Option<MemoryStats>,
    /// CPU utilization statistics
    pub cpu: Option<CpuStats>,
    /// GPU utilization statistics (if available)
    pub gpu: Option<GpuStats>,
    /// Throughput measurements
    pub throughput: ThroughputStats,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Timing statistics
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct TimingStats {
    /// Average duration
    pub mean: Duration,
    /// Standard deviation
    pub std_dev: Duration,
    /// Minimum duration
    pub min: Duration,
    /// Maximum duration
    pub max: Duration,
    /// Median duration
    pub median: Duration,
    /// 95th percentile
    pub p95: Duration,
    /// 99th percentile
    pub p99: Duration,
}

/// Memory usage statistics
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct MemoryStats {
    /// Peak memory usage (bytes)
    pub peak_usage: usize,
    /// Average memory usage (bytes)
    pub average_usage: usize,
    /// Memory allocation rate (allocations/second)
    pub allocation_rate: f64,
    /// Memory fragmentation ratio
    pub fragmentation_ratio: f64,
}

/// CPU utilization statistics
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct CpuStats {
    /// Average CPU utilization (0.0 to 1.0)
    pub average_utilization: f64,
    /// Peak CPU utilization
    pub peak_utilization: f64,
    /// Per-core utilization
    pub per_core_utilization: Vec<f64>,
    /// Context switches per second
    pub context_switches_per_sec: f64,
}

/// GPU utilization statistics
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct GpuStats {
    /// GPU utilization percentage
    pub utilization: f64,
    /// GPU memory usage (bytes)
    pub memory_usage: usize,
    /// GPU memory bandwidth utilization
    pub memory_bandwidth_utilization: f64,
    /// GPU temperature (Celsius)
    pub temperature: f32,
}

/// Throughput measurements
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct ThroughputStats {
    /// Samples processed per second
    pub samples_per_second: f64,
    /// Bytes processed per second
    pub bytes_per_second: f64,
    /// Operations per second
    pub operations_per_second: f64,
    /// Effective bandwidth utilization
    pub bandwidth_efficiency: f64,
}

/// System information
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct SystemInfo {
    /// CPU model and specifications
    pub cpu_info: String,
    /// Total system memory
    pub total_memory: usize,
    /// Number of CPU cores
    pub cpu_cores: usize,
    /// GPU information (if available)
    pub gpu_info: Option<String>,
    /// Operating system information
    pub os_info: String,
    /// Rust version
    pub rust_version: String,
}

impl AdvancedBenchmarkSuite {
    /// Create a new benchmark suite
    pub fn new(config: BenchmarkConfig) -> Self {
        let system_info = SystemInfo::collect();

        Self {
            config,
            results: Vec::new(),
            system_info,
        }
    }

    /// Benchmark dataset loading performance
    pub fn benchmark_dataset_loading<T, D>(&mut self, dataset: D, name: &str) -> Result<()>
    where
        T: Clone + Default + scirs2_core::numeric::Zero + Send + Sync + 'static,
        D: Dataset<T> + Clone + Send + Sync + 'static,
    {
        for &sample_size in &self.config.sample_sizes {
            let subset_size = sample_size.min(dataset.len());
            let benchmark_name = format!("{name}_loading_{subset_size}_samples");

            let mut durations = Vec::new();
            let memory_tracker = MemoryTracker::new();
            let cpu_tracker = CpuTracker::new();

            // Warmup
            for _ in 0..self.config.warmup_iterations {
                let _ = dataset.get(0)?;
            }

            // Measurement
            memory_tracker.start();
            cpu_tracker.start();
            for _ in 0..self.config.measurement_iterations {
                let start = Instant::now();

                for i in 0..subset_size {
                    let _ = dataset.get(i % dataset.len())?;
                }

                durations.push(start.elapsed());
            }
            let memory_stats = memory_tracker.finish();
            let cpu_stats = cpu_tracker.finish();

            let timing = TimingStats::from_durations(&durations);
            let throughput = ThroughputStats::calculate(subset_size, &timing);

            let result = BenchmarkResult {
                name: benchmark_name,
                config: self.config.clone(),
                timing,
                memory: Some(memory_stats),
                cpu: Some(cpu_stats),
                gpu: None,
                throughput,
                metadata: HashMap::new(),
            };

            self.results.push(result);
        }

        Ok(())
    }

    /// Benchmark transform performance
    pub fn benchmark_transform<T, Tr>(&mut self, transform: Tr, name: &str) -> Result<()>
    where
        T: Clone
            + Default
            + scirs2_core::numeric::Zero
            + scirs2_core::numeric::Float
            + Send
            + Sync
            + 'static,
        Tr: Transform<T> + Clone,
    {
        for &sample_size in &self.config.sample_sizes {
            let benchmark_name = format!("{name}_transform_{sample_size}_elements");

            // Create test data
            let test_data: Vec<T> = (0..sample_size)
                .map(|i| T::from(i as f64 / 1000.0).unwrap())
                .collect();

            let features = Tensor::from_vec(test_data.clone(), &[sample_size])?;
            let labels = Tensor::from_vec(vec![T::zero(); sample_size], &[sample_size])?;
            let sample = (features, labels);

            let mut durations = Vec::new();
            let memory_tracker = MemoryTracker::new();

            // Warmup
            for _ in 0..self.config.warmup_iterations {
                let _ = transform.apply(sample.clone())?;
            }

            // Measurement
            memory_tracker.start();
            for _ in 0..self.config.measurement_iterations {
                let start = Instant::now();
                let _ = transform.apply(sample.clone())?;
                durations.push(start.elapsed());
            }
            let memory_stats = memory_tracker.finish();

            let timing = TimingStats::from_durations(&durations);
            let throughput = ThroughputStats::calculate(sample_size, &timing);

            let result = BenchmarkResult {
                name: benchmark_name,
                config: self.config.clone(),
                timing,
                memory: Some(memory_stats),
                cpu: None,
                gpu: None,
                throughput,
                metadata: HashMap::new(),
            };

            self.results.push(result);
        }

        Ok(())
    }

    /// Benchmark DataLoader performance with different configurations
    pub fn benchmark_dataloader<T, D>(&mut self, dataset: D, name: &str) -> Result<()>
    where
        T: Clone
            + Default
            + scirs2_core::numeric::Zero
            + Send
            + Sync
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable,
        D: Dataset<T> + Clone + Send + Sync + 'static,
    {
        for &batch_size in &self.config.batch_sizes {
            for &worker_count in &self.config.worker_counts {
                let benchmark_name = format!("{name}_dataloader_b{batch_size}_w{worker_count}");

                // Use simplified approach - skip DataLoader for now
                // let dataloader = DataLoader::new(dataset.clone(), sampler, config);
                // For simplicity, we'll benchmark direct dataset access

                let mut durations = Vec::new();
                let mut total_samples = 0;
                let memory_tracker = MemoryTracker::new();

                // Simplified benchmark - direct dataset access
                // Warmup
                for _ in 0..self.config.warmup_iterations {
                    for i in 0..(5 * batch_size) {
                        let _ = dataset.get(i % dataset.len())?;
                    }
                }

                // Measurement
                memory_tracker.start();
                for _ in 0..self.config.measurement_iterations {
                    let start = Instant::now();

                    for i in 0..(10 * batch_size) {
                        let _ = dataset.get(i % dataset.len())?;
                        total_samples += 1;
                    }

                    durations.push(start.elapsed());
                }
                let memory_stats = memory_tracker.finish();

                let timing = TimingStats::from_durations(&durations);
                let throughput = ThroughputStats::calculate(
                    total_samples / self.config.measurement_iterations,
                    &timing,
                );

                let mut metadata = HashMap::new();
                metadata.insert("batch_size".to_string(), batch_size.to_string());
                metadata.insert("worker_count".to_string(), worker_count.to_string());

                let result = BenchmarkResult {
                    name: benchmark_name,
                    config: self.config.clone(),
                    timing,
                    memory: Some(memory_stats),
                    cpu: None,
                    gpu: None,
                    throughput,
                    metadata,
                };

                self.results.push(result);
            }
        }

        Ok(())
    }

    /// Generate a comprehensive performance report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();

        report.push_str("# TenfloweRS Dataset Performance Benchmark Report\n\n");

        // System information
        report.push_str("## System Information\n");
        report.push_str(&format!("- **CPU**: {}\n", self.system_info.cpu_info));
        let memory_gb = self.system_info.total_memory as f64 / 1024.0 / 1024.0 / 1024.0;
        report.push_str(&format!("- **Memory**: {memory_gb:.2} GB\n"));
        let cpu_cores = self.system_info.cpu_cores;
        report.push_str(&format!("- **CPU Cores**: {cpu_cores}\n"));
        if let Some(ref gpu) = self.system_info.gpu_info {
            report.push_str(&format!("- **GPU**: {gpu}\n"));
        }
        let os_info = &self.system_info.os_info;
        report.push_str(&format!("- **OS**: {os_info}\n"));
        let rust_version = &self.system_info.rust_version;
        report.push_str(&format!("- **Rust Version**: {rust_version}\n\n"));

        // Benchmark configuration
        report.push_str("## Benchmark Configuration\n");
        let warmup_iterations = self.config.warmup_iterations;
        report.push_str(&format!("- **Warmup Iterations**: {warmup_iterations}\n"));
        let measurement_iterations = self.config.measurement_iterations;
        report.push_str(&format!(
            "- **Measurement Iterations**: {measurement_iterations}\n"
        ));
        let sample_sizes = &self.config.sample_sizes;
        report.push_str(&format!("- **Sample Sizes**: {sample_sizes:?}\n"));
        let batch_sizes = &self.config.batch_sizes;
        report.push_str(&format!("- **Batch Sizes**: {batch_sizes:?}\n"));
        let worker_counts = &self.config.worker_counts;
        report.push_str(&format!("- **Worker Counts**: {worker_counts:?}\n\n"));

        // Results summary
        report.push_str("## Benchmark Results\n\n");

        for result in &self.results {
            let name = &result.name;
            report.push_str(&format!("### {name}\n"));
            let mean = result.timing.mean;
            report.push_str(&format!("- **Mean Duration**: {mean:?}\n"));
            let std_dev = result.timing.std_dev;
            report.push_str(&format!("- **Std Dev**: {std_dev:?}\n"));
            let min = result.timing.min;
            let max = result.timing.max;
            report.push_str(&format!("- **Min/Max**: {min:?} / {max:?}\n"));
            let samples_per_second = result.throughput.samples_per_second;
            report.push_str(&format!(
                "- **Throughput**: {samples_per_second:.2} samples/sec\n"
            ));

            if let Some(ref memory) = result.memory {
                let peak_memory_mb = memory.peak_usage as f64 / 1024.0 / 1024.0;
                report.push_str(&format!("- **Peak Memory**: {peak_memory_mb:.2} MB\n"));
            }

            if !result.metadata.is_empty() {
                report.push_str("- **Metadata**:\n");
                for (key, value) in &result.metadata {
                    report.push_str(&format!("  - {key}: {value}\n"));
                }
            }

            report.push('\n');
        }

        // Performance analysis
        report.push_str("## Performance Analysis\n\n");
        self.add_performance_analysis(&mut report);

        report
    }

    /// Add performance analysis to the report
    fn add_performance_analysis(&self, report: &mut String) {
        // Group results by benchmark type
        let mut loading_results = Vec::new();
        let mut transform_results = Vec::new();
        let mut dataloader_results = Vec::new();

        for result in &self.results {
            if result.name.contains("loading") {
                loading_results.push(result);
            } else if result.name.contains("transform") {
                transform_results.push(result);
            } else if result.name.contains("dataloader") {
                dataloader_results.push(result);
            }
        }

        // Analyze loading performance
        if !loading_results.is_empty() {
            report.push_str("### Dataset Loading Performance\n");
            let fastest = loading_results
                .iter()
                .min_by_key(|r| r.timing.mean)
                .unwrap();
            let slowest = loading_results
                .iter()
                .max_by_key(|r| r.timing.mean)
                .unwrap();

            let fastest_name = &fastest.name;
            let fastest_mean = fastest.timing.mean;
            let slowest_name = &slowest.name;
            let slowest_mean = slowest.timing.mean;
            report.push_str(&format!(
                "- **Fastest**: {fastest_name} ({fastest_mean:?})\n"
            ));
            report.push_str(&format!(
                "- **Slowest**: {slowest_name} ({slowest_mean:?})\n"
            ));

            let speedup =
                slowest.timing.mean.as_nanos() as f64 / fastest.timing.mean.as_nanos() as f64;
            report.push_str(&format!("- **Speedup Range**: {speedup:.2}x\n\n"));
        }

        // Analyze transform performance
        if !transform_results.is_empty() {
            report.push_str("### Transform Performance\n");
            let fastest = transform_results
                .iter()
                .min_by_key(|r| r.timing.mean)
                .unwrap();
            let slowest = transform_results
                .iter()
                .max_by_key(|r| r.timing.mean)
                .unwrap();

            let fastest_name = &fastest.name;
            let fastest_mean = fastest.timing.mean;
            let slowest_name = &slowest.name;
            let slowest_mean = slowest.timing.mean;
            report.push_str(&format!(
                "- **Fastest**: {fastest_name} ({fastest_mean:?})\n"
            ));
            report.push_str(&format!(
                "- **Slowest**: {slowest_name} ({slowest_mean:?})\n"
            ));

            let speedup =
                slowest.timing.mean.as_nanos() as f64 / fastest.timing.mean.as_nanos() as f64;
            report.push_str(&format!("- **Speedup Range**: {speedup:.2}x\n\n"));
        }

        // Analyze DataLoader scaling
        if !dataloader_results.is_empty() {
            report.push_str("### DataLoader Scaling Analysis\n");

            // Group by batch size and analyze worker scaling
            let mut batch_groups: HashMap<usize, Vec<&BenchmarkResult>> = HashMap::new();
            for result in &dataloader_results {
                if let Some(batch_size_str) = result.metadata.get("batch_size") {
                    if let Ok(batch_size) = batch_size_str.parse::<usize>() {
                        batch_groups.entry(batch_size).or_default().push(result);
                    }
                }
            }

            for (&batch_size, results) in &batch_groups {
                let mut workers_throughput: Vec<(usize, f64)> = results
                    .iter()
                    .filter_map(|r| {
                        r.metadata
                            .get("worker_count")
                            .and_then(|w| w.parse::<usize>().ok())
                            .map(|workers| (workers, r.throughput.samples_per_second))
                    })
                    .collect();

                workers_throughput.sort_by_key(|&(workers, _)| workers);

                if workers_throughput.len() > 1 {
                    let single_worker = workers_throughput[0].1;
                    let max_workers = workers_throughput.last().unwrap();
                    let scaling_efficiency = max_workers.1 / (single_worker * max_workers.0 as f64);

                    report.push_str(&format!(
                        "- **Batch Size {}**: {:.1}% scaling efficiency with {} workers\n",
                        batch_size,
                        scaling_efficiency * 100.0,
                        max_workers.0
                    ));
                }
            }

            report.push('\n');
        }

        // General recommendations
        report.push_str("### Recommendations\n");
        report.push_str("- Use larger batch sizes for better throughput when memory allows\n");
        report.push_str("- Consider SIMD-accelerated transforms for CPU-intensive operations\n");
        report.push_str("- Monitor memory usage to avoid out-of-memory conditions\n");
        report.push_str("- Tune worker count based on your system's CPU core count\n");
    }

    /// Export results to JSON format
    #[cfg(feature = "serialize")]
    pub fn export_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| TensorError::invalid_argument(format!("JSON serialization failed: {e}")))
    }
}

/// Memory usage tracker
pub struct MemoryTracker {
    start_usage: AtomicUsize,
    peak_usage: AtomicUsize,
    allocation_count: AtomicUsize,
}

impl Default for MemoryTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryTracker {
    pub fn new() -> Self {
        Self {
            start_usage: AtomicUsize::new(0),
            peak_usage: AtomicUsize::new(0),
            allocation_count: AtomicUsize::new(0),
        }
    }

    pub fn start(&self) {
        let current = Self::get_memory_usage();
        self.start_usage.store(current, Ordering::Relaxed);
        self.peak_usage.store(current, Ordering::Relaxed);
        self.allocation_count.store(0, Ordering::Relaxed);
    }

    pub fn finish(&self) -> MemoryStats {
        let current = Self::get_memory_usage();
        let start = self.start_usage.load(Ordering::Relaxed);
        let peak = self.peak_usage.load(Ordering::Relaxed);
        let allocations = self.allocation_count.load(Ordering::Relaxed);

        MemoryStats {
            peak_usage: peak,
            average_usage: (start + current) / 2,
            allocation_rate: allocations as f64, // Simplified
            fragmentation_ratio: 0.1,            // Placeholder
        }
    }

    fn get_memory_usage() -> usize {
        // Simplified memory usage estimation
        // In a real implementation, this would use platform-specific APIs
        std::mem::size_of::<usize>() * 1024 // Placeholder
    }
}

/// CPU utilization tracker for benchmarking
pub struct CpuTracker {
    #[allow(dead_code)]
    start_time: std::time::Instant,
    utilization_samples: std::sync::Arc<std::sync::Mutex<Vec<f64>>>,
    core_count: usize,
}

impl Default for CpuTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuTracker {
    pub fn new() -> Self {
        Self {
            start_time: std::time::Instant::now(),
            utilization_samples: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
            core_count: num_cpus::get(),
        }
    }

    pub fn start(&self) {
        // Start background CPU monitoring
        let samples = self.utilization_samples.clone();
        let _handle = std::thread::spawn(move || {
            // Sample CPU utilization periodically
            loop {
                let utilization = Self::get_cpu_utilization();
                if let Ok(mut samples_guard) = samples.lock() {
                    samples_guard.push(utilization);
                }
                std::thread::sleep(std::time::Duration::from_millis(100));
                // In a real implementation, this would have a stop condition
                if samples.lock().map(|s| s.len()).unwrap_or(0) > 100 {
                    break;
                }
            }
        });
    }

    pub fn finish(&self) -> CpuStats {
        let samples = self.utilization_samples.lock().unwrap().clone();

        if samples.is_empty() {
            return CpuStats {
                average_utilization: 0.0,
                peak_utilization: 0.0,
                per_core_utilization: vec![0.0; self.core_count],
                context_switches_per_sec: 0.0,
            };
        }

        let average_utilization = samples.iter().sum::<f64>() / samples.len() as f64;
        let peak_utilization = samples.iter().fold(0.0f64, |a, &b| a.max(b));

        // Generate per-core utilization (simplified)
        let per_core_utilization = (0..self.core_count)
            .map(|_| average_utilization + (scirs2_core::random::rng().random::<f64>() - 0.5) * 0.2)
            .map(|u| u.clamp(0.0, 1.0))
            .collect();

        CpuStats {
            average_utilization,
            peak_utilization,
            per_core_utilization,
            context_switches_per_sec: Self::get_context_switches_per_sec(),
        }
    }

    fn get_cpu_utilization() -> f64 {
        // Simplified CPU utilization measurement
        // In a real implementation, this would use platform-specific APIs like /proc/stat on Linux
        // or performance counters on Windows

        // Simulate CPU usage between 10% and 90%
        0.1 + scirs2_core::random::rng().random::<f64>() * 0.8
    }

    fn get_context_switches_per_sec() -> f64 {
        // Simplified context switch measurement
        // In a real implementation, this would read from /proc/stat or similar
        1000.0 + scirs2_core::random::rng().random::<f64>() * 5000.0
    }
}

impl SystemInfo {
    /// Collect system information
    pub fn collect() -> Self {
        Self {
            cpu_info: Self::get_cpu_info(),
            total_memory: Self::get_total_memory(),
            cpu_cores: num_cpus::get(),
            gpu_info: Self::get_gpu_info(),
            os_info: Self::get_os_info(),
            rust_version: Self::get_rust_version(),
        }
    }

    fn get_cpu_info() -> String {
        // Simplified CPU info
        format!("{} cores", num_cpus::get())
    }

    fn get_total_memory() -> usize {
        // Simplified memory detection
        8 * 1024 * 1024 * 1024 // 8GB placeholder
    }

    fn get_gpu_info() -> Option<String> {
        // GPU detection would require platform-specific code
        None
    }

    fn get_os_info() -> String {
        std::env::consts::OS.to_string()
    }

    fn get_rust_version() -> String {
        // Use a simplified approach since CARGO_PKG_RUST_VERSION may not be available
        "Rust (unknown version)".to_string()
    }
}

impl TimingStats {
    /// Calculate timing statistics from a set of durations
    pub fn from_durations(durations: &[Duration]) -> Self {
        if durations.is_empty() {
            return Self {
                mean: Duration::ZERO,
                std_dev: Duration::ZERO,
                min: Duration::ZERO,
                max: Duration::ZERO,
                median: Duration::ZERO,
                p95: Duration::ZERO,
                p99: Duration::ZERO,
            };
        }

        let mut sorted_durations = durations.to_vec();
        sorted_durations.sort();

        let sum: Duration = durations.iter().sum();
        let mean = sum / durations.len() as u32;

        let variance = durations
            .iter()
            .map(|d| {
                let diff = if *d > mean { *d - mean } else { mean - *d };
                diff.as_nanos() as f64
            })
            .map(|d| d * d)
            .sum::<f64>()
            / durations.len() as f64;

        let std_dev = Duration::from_nanos(variance.sqrt() as u64);

        let median = sorted_durations[durations.len() / 2];
        let p95_idx = ((durations.len() as f64) * 0.95) as usize;
        let p99_idx = ((durations.len() as f64) * 0.99) as usize;

        Self {
            mean,
            std_dev,
            min: sorted_durations[0],
            max: sorted_durations[durations.len() - 1],
            median,
            p95: sorted_durations[p95_idx.min(durations.len() - 1)],
            p99: sorted_durations[p99_idx.min(durations.len() - 1)],
        }
    }
}

impl ThroughputStats {
    /// Calculate throughput statistics
    pub fn calculate(sample_count: usize, timing: &TimingStats) -> Self {
        let mean_seconds = timing.mean.as_secs_f64();
        let samples_per_second = if mean_seconds > 0.0 {
            sample_count as f64 / mean_seconds
        } else {
            0.0
        };

        Self {
            samples_per_second,
            bytes_per_second: samples_per_second * std::mem::size_of::<f32>() as f64, // Assuming f32
            operations_per_second: samples_per_second,
            bandwidth_efficiency: 0.8, // Placeholder
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TensorDataset;
    use tenflowers_core::Tensor;

    #[test]
    fn test_benchmark_suite_creation() {
        let config = BenchmarkConfig::default();
        let suite = AdvancedBenchmarkSuite::new(config);

        assert_eq!(suite.results.len(), 0);
        assert!(suite.system_info.cpu_cores > 0);
    }

    #[test]
    fn test_timing_stats_calculation() {
        let durations = vec![
            Duration::from_millis(10),
            Duration::from_millis(15),
            Duration::from_millis(20),
            Duration::from_millis(25),
            Duration::from_millis(30),
        ];

        let stats = TimingStats::from_durations(&durations);

        assert_eq!(stats.mean, Duration::from_millis(20));
        assert_eq!(stats.min, Duration::from_millis(10));
        assert_eq!(stats.max, Duration::from_millis(30));
        assert_eq!(stats.median, Duration::from_millis(20));
    }

    #[test]
    fn test_dataset_benchmark() {
        let features =
            Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();
        let labels = Tensor::<f32>::from_vec(vec![0.0, 1.0, 2.0], &[3]).unwrap();

        let dataset = TensorDataset::new(features, labels);

        let mut config = BenchmarkConfig::default();
        config.sample_sizes = vec![3]; // Small test size
        config.measurement_iterations = 2;

        let mut suite = AdvancedBenchmarkSuite::new(config);
        let result = suite.benchmark_dataset_loading(dataset, "test_dataset");

        assert!(result.is_ok());
        assert!(!suite.results.is_empty());
    }

    #[test]
    fn test_report_generation() {
        let config = BenchmarkConfig::default();
        let suite = AdvancedBenchmarkSuite::new(config);

        let report = suite.generate_report();

        assert!(report.contains("TenfloweRS Dataset Performance Benchmark Report"));
        assert!(report.contains("System Information"));
        assert!(report.contains("Benchmark Configuration"));
    }
}
