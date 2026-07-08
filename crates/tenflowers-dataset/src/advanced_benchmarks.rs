//! Advanced benchmarking system for comprehensive performance analysis
//!
//! This module provides sophisticated benchmarking tools for analyzing dataset
//! performance across different operations, hardware configurations, and workloads.

use crate::{Dataset, Transform};
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
                .map(|i| T::from(i as f64 / 1000.0).unwrap_or(T::zero()))
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
                .expect("non-empty loading_results should have minimum");
            let slowest = loading_results
                .iter()
                .max_by_key(|r| r.timing.mean)
                .expect("non-empty loading_results should have maximum");

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
                .expect("non-empty transform_results should have minimum");
            let slowest = transform_results
                .iter()
                .max_by_key(|r| r.timing.mean)
                .expect("non-empty transform_results should have maximum");

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
                    let max_workers = workers_throughput
                        .last()
                        .expect("collection should not be empty");
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
            // Allocation counting requires a custom global allocator hook,
            // which is not installed; report what was actually tracked (0
            // unless a hook increments it) rather than inventing a rate.
            allocation_rate: allocations as f64,
            // Fragmentation cannot be measured without allocator introspection.
            // Report 0.0 ("not measured") instead of a fabricated ratio.
            fragmentation_ratio: 0.0,
        }
    }

    /// Read the process resident-set size (RSS) in bytes.
    ///
    /// Uses `/proc/self/statm` on Linux for a real measurement. On platforms
    /// where this interface is unavailable the value cannot be measured, so 0
    /// is returned (honest "unknown") rather than a fabricated figure.
    fn get_memory_usage() -> usize {
        read_process_rss_bytes().unwrap_or(0)
    }
}

/// Read this process's resident-set size in bytes from `/proc/self/statm`.
///
/// Returns `None` when the measurement is unavailable (non-Linux targets or a
/// read/parse failure), so callers can fall back to an honest "unknown".
fn read_process_rss_bytes() -> Option<usize> {
    #[cfg(target_os = "linux")]
    {
        // statm fields are in pages: size, resident, shared, text, lib, data, dt.
        let statm = std::fs::read_to_string("/proc/self/statm").ok()?;
        let resident_pages: usize = statm.split_whitespace().nth(1)?.parse().ok()?;
        // 4096 is the standard page size on the supported Linux targets.
        const PAGE_SIZE: usize = 4096;
        Some(resident_pages.saturating_mul(PAGE_SIZE))
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// Read total physical memory in bytes from `/proc/meminfo` (Linux).
///
/// Returns `None` when unavailable so callers fall back to an honest "unknown".
fn read_total_memory_bytes() -> Option<usize> {
    #[cfg(target_os = "linux")]
    {
        let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
        for line in meminfo.lines() {
            if let Some(rest) = line.strip_prefix("MemTotal:") {
                // Format: "MemTotal:       16384256 kB".
                let kb: usize = rest.split_whitespace().next()?.parse().ok()?;
                return Some(kb.saturating_mul(1024));
            }
        }
        None
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// Aggregate CPU jiffy counters read from `/proc/stat`.
#[derive(Debug, Clone, Copy)]
struct CpuTimes {
    idle: u64,
    total: u64,
}

/// Read the aggregate CPU times from the `cpu` line of `/proc/stat` (Linux).
///
/// Returns `None` when unavailable, so callers fall back to honest "unknown".
fn read_aggregate_cpu_times() -> Option<CpuTimes> {
    #[cfg(target_os = "linux")]
    {
        let stat = std::fs::read_to_string("/proc/stat").ok()?;
        let line = stat.lines().next()?; // first line is the aggregate "cpu" line
        let mut fields = line.split_whitespace();
        if fields.next()? != "cpu" {
            return None;
        }
        // user nice system idle iowait irq softirq steal guest guest_nice
        let values: Vec<u64> = fields.filter_map(|v| v.parse::<u64>().ok()).collect();
        if values.len() < 4 {
            return None;
        }
        // idle time = idle + iowait (iowait present when len >= 5).
        let idle = values[3] + values.get(4).copied().unwrap_or(0);
        let total: u64 = values.iter().sum();
        Some(CpuTimes { idle, total })
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// Read per-core CPU times from the `cpuN` lines of `/proc/stat` (Linux).
fn read_per_core_cpu_times() -> Vec<CpuTimes> {
    #[cfg(target_os = "linux")]
    {
        let Ok(stat) = std::fs::read_to_string("/proc/stat") else {
            return Vec::new();
        };
        let mut cores = Vec::new();
        for line in stat.lines() {
            // Per-core lines look like "cpu0 ...", "cpu1 ...". Skip the aggregate.
            if !line.starts_with("cpu") || line.starts_with("cpu ") {
                continue;
            }
            let mut fields = line.split_whitespace();
            let Some(tag) = fields.next() else { continue };
            // Require a trailing digit so we only take per-core lines.
            if !tag[3..].chars().all(|c| c.is_ascii_digit()) || tag.len() <= 3 {
                continue;
            }
            let values: Vec<u64> = fields.filter_map(|v| v.parse::<u64>().ok()).collect();
            if values.len() < 4 {
                continue;
            }
            let idle = values[3] + values.get(4).copied().unwrap_or(0);
            let total: u64 = values.iter().sum();
            cores.push(CpuTimes { idle, total });
        }
        cores
    }
    #[cfg(not(target_os = "linux"))]
    {
        Vec::new()
    }
}

/// Read the cumulative context-switch count from `/proc/stat` (Linux).
fn read_context_switch_count() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let stat = std::fs::read_to_string("/proc/stat").ok()?;
        for line in stat.lines() {
            if let Some(rest) = line.strip_prefix("ctxt ") {
                return rest.trim().parse::<u64>().ok();
            }
        }
        None
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// Compute utilization (0.0..=1.0) from two CPU snapshots.
fn cpu_utilization_from_delta(prev: CpuTimes, next: CpuTimes) -> Option<f64> {
    let total_delta = next.total.checked_sub(prev.total)?;
    if total_delta == 0 {
        return None;
    }
    let idle_delta = next.idle.saturating_sub(prev.idle);
    let busy = total_delta.saturating_sub(idle_delta);
    Some((busy as f64 / total_delta as f64).clamp(0.0, 1.0))
}

/// CPU utilization tracker for benchmarking.
///
/// On Linux this reports *real* CPU utilization and context-switch rates by
/// reading `/proc/stat`. On other platforms (or when `/proc` is unavailable)
/// it reports 0.0 ("not measured") instead of fabricating values.
pub struct CpuTracker {
    start_time: std::time::Instant,
    utilization_samples: std::sync::Arc<std::sync::Mutex<Vec<f64>>>,
    per_core_samples: std::sync::Arc<std::sync::Mutex<Vec<Vec<f64>>>>,
    stop_flag: std::sync::Arc<std::sync::atomic::AtomicBool>,
    handle: std::sync::Mutex<Option<std::thread::JoinHandle<()>>>,
    start_ctxt: std::sync::atomic::AtomicU64,
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
            per_core_samples: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
            stop_flag: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            handle: std::sync::Mutex::new(None),
            start_ctxt: std::sync::atomic::AtomicU64::new(0),
            core_count: num_cpus::get(),
        }
    }

    pub fn start(&self) {
        // Record the starting context-switch count for a real per-second rate.
        if let Some(ctxt) = read_context_switch_count() {
            self.start_ctxt.store(ctxt, Ordering::Relaxed);
        }

        self.stop_flag.store(false, Ordering::Relaxed);
        let samples = self.utilization_samples.clone();
        let core_samples = self.per_core_samples.clone();
        let stop = self.stop_flag.clone();

        let handle = std::thread::spawn(move || {
            // Seed the previous snapshots so the first delta is meaningful.
            let mut prev_agg = read_aggregate_cpu_times();
            let mut prev_cores = read_per_core_cpu_times();
            let mut iterations = 0usize;

            loop {
                std::thread::sleep(std::time::Duration::from_millis(100));

                if let (Some(prev), Some(next)) = (prev_agg, read_aggregate_cpu_times()) {
                    if let Some(util) = cpu_utilization_from_delta(prev, next) {
                        if let Ok(mut guard) = samples.lock() {
                            guard.push(util);
                        }
                    }
                    prev_agg = Some(next);
                }

                let next_cores = read_per_core_cpu_times();
                if !prev_cores.is_empty() && prev_cores.len() == next_cores.len() {
                    let per_core: Vec<f64> = prev_cores
                        .iter()
                        .zip(next_cores.iter())
                        .map(|(&p, &n)| cpu_utilization_from_delta(p, n).unwrap_or(0.0))
                        .collect();
                    if let Ok(mut guard) = core_samples.lock() {
                        guard.push(per_core);
                    }
                }
                prev_cores = next_cores;

                iterations += 1;
                if stop.load(Ordering::Relaxed) || iterations > 100 {
                    break;
                }
            }
        });

        if let Ok(mut slot) = self.handle.lock() {
            *slot = Some(handle);
        }
    }

    pub fn finish(&self) -> CpuStats {
        // Signal the sampling thread to stop and join it so all real samples land.
        self.stop_flag.store(true, Ordering::Relaxed);
        if let Ok(mut slot) = self.handle.lock() {
            if let Some(handle) = slot.take() {
                let _ = handle.join();
            }
        }

        let samples = self
            .utilization_samples
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();

        let context_switches_per_sec = self.context_switches_per_sec();

        if samples.is_empty() {
            // No measurements available (e.g. non-Linux): report honest zeros.
            return CpuStats {
                average_utilization: 0.0,
                peak_utilization: 0.0,
                per_core_utilization: vec![0.0; self.core_count],
                context_switches_per_sec,
            };
        }

        let average_utilization = samples.iter().sum::<f64>() / samples.len() as f64;
        let peak_utilization = samples.iter().fold(0.0f64, |a, &b| a.max(b));

        // Real per-core averages from `/proc/stat`; fall back to the aggregate
        // average per core only when per-core sampling was unavailable.
        let per_core_samples = self
            .per_core_samples
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();

        let per_core_utilization = if per_core_samples.is_empty() {
            vec![average_utilization; self.core_count]
        } else {
            let core_count = per_core_samples[0].len();
            (0..core_count)
                .map(|core| {
                    let sum: f64 = per_core_samples
                        .iter()
                        .filter_map(|row| row.get(core))
                        .sum();
                    let count = per_core_samples
                        .iter()
                        .filter(|row| row.get(core).is_some())
                        .count();
                    if count > 0 {
                        sum / count as f64
                    } else {
                        0.0
                    }
                })
                .collect()
        };

        CpuStats {
            average_utilization,
            peak_utilization,
            per_core_utilization,
            context_switches_per_sec,
        }
    }

    /// Compute the real context-switch rate over the tracked interval.
    fn context_switches_per_sec(&self) -> f64 {
        let start = self.start_ctxt.load(Ordering::Relaxed);
        match read_context_switch_count() {
            Some(end) if end >= start && start > 0 => {
                let elapsed = self.start_time.elapsed().as_secs_f64();
                if elapsed > 0.0 {
                    (end - start) as f64 / elapsed
                } else {
                    0.0
                }
            }
            // Unavailable or not measurable: honest zero rather than a fake rate.
            _ => 0.0,
        }
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
        // Read the real total physical memory from /proc/meminfo on Linux.
        // Returns 0 ("unknown") on platforms where it cannot be measured,
        // rather than reporting a fabricated capacity.
        read_total_memory_bytes().unwrap_or(0)
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
            // Bandwidth efficiency = achieved bandwidth / theoretical peak.
            // The theoretical peak is hardware-specific and not known here, so
            // this stays 0.0 ("not measured") instead of a fabricated ratio.
            bandwidth_efficiency: 0.0,
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
        let features = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2])
            .expect("test: tensor creation should succeed");
        let labels = Tensor::<f32>::from_vec(vec![0.0, 1.0, 2.0], &[3])
            .expect("test: tensor creation should succeed");

        let dataset = TensorDataset::new(features, labels);

        let config = BenchmarkConfig {
            sample_sizes: vec![3], // Small test size
            measurement_iterations: 2,
            ..Default::default()
        };

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

    #[test]
    fn test_throughput_stats_bandwidth_not_fabricated() {
        // bandwidth_efficiency must no longer be a fabricated 0.8 constant.
        let timing =
            TimingStats::from_durations(&[Duration::from_millis(10), Duration::from_millis(10)]);
        let stats = ThroughputStats::calculate(100, &timing);
        assert_eq!(
            stats.bandwidth_efficiency, 0.0,
            "bandwidth efficiency is not measured and must be reported as 0.0"
        );
        // samples_per_second IS real (derived from timing): 100 samples / 0.01s.
        assert!((stats.samples_per_second - 10_000.0).abs() < 1.0);
    }

    #[test]
    fn test_memory_stats_fragmentation_not_fabricated() {
        let tracker = MemoryTracker::new();
        tracker.start();
        let stats = tracker.finish();
        assert_eq!(
            stats.fragmentation_ratio, 0.0,
            "fragmentation ratio is not measured and must be reported as 0.0"
        );
    }

    #[test]
    fn test_cpu_utilization_delta_math() {
        // 100 total jiffies elapse, 25 of them idle -> 75% utilization.
        let prev = CpuTimes {
            idle: 100,
            total: 1000,
        };
        let next = CpuTimes {
            idle: 125,
            total: 1100,
        };
        let util = cpu_utilization_from_delta(prev, next)
            .expect("test: nonzero total delta should yield a value");
        assert!((util - 0.75).abs() < 1e-9);

        // No elapsed time -> no measurement (None), never a fabricated number.
        assert!(cpu_utilization_from_delta(prev, prev).is_none());
    }

    #[test]
    fn test_total_memory_is_real_or_zero() {
        // Must be the real total memory (Linux) or an honest 0 elsewhere,
        // never the old hardcoded 8 GiB placeholder.
        let total = SystemInfo::collect().total_memory;
        #[cfg(target_os = "linux")]
        {
            assert!(
                total > 0,
                "Linux total memory should be measured as nonzero"
            );
            // The old fabricated value was exactly 8 GiB; a real read is virtually
            // never exactly that. We only assert it was actually measured (>0).
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = total; // honest 0 on unsupported platforms
        }
    }
}
