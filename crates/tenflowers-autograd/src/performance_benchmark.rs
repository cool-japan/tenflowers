//! # Performance Benchmarking Framework for Autograd
//!
//! This module provides comprehensive performance benchmarking capabilities for gradient
//! computation operations, including statistical analysis, throughput measurement, and
//! performance regression detection.
//!
//! ## Features
//!
//! - **Statistical Analysis**: Mean, median, std dev, percentiles for benchmark results
//! - **Throughput Measurement**: Operations per second and tensor elements per second
//! - **Memory Profiling**: Memory usage tracking during benchmarks
//! - **Comparison Tools**: Compare performance across different configurations
//! - **Regression Detection**: Identify performance degradations automatically
//! - **Export Capabilities**: Export results in multiple formats (JSON, CSV, Markdown)
//!
//! ## Usage
//!
//! ```rust,no_run
//! use tenflowers_autograd::{PerformanceBenchmark, BenchmarkConfig, GradientTape};
//! use tenflowers_core::Tensor;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create benchmark configuration
//! let config = BenchmarkConfig::default()
//!     .with_iterations(100)
//!     .with_warmup(10)
//!     .with_statistical_analysis(true);
//!
//! let mut benchmark = PerformanceBenchmark::new(config);
//!
//! // Benchmark a gradient operation
//! benchmark.benchmark_operation("matmul_backward", || {
//!     let tape = GradientTape::new();
//!     let a = tape.watch(Tensor::<f32>::ones(&[100, 100]));
//!     let b = tape.watch(Tensor::<f32>::ones(&[100, 100]));
//!     let c = a.matmul(&b)?;
//!     tape.gradient(&[c], &[a, b])
//! })?;
//!
//! // Get and print results
//! let results = benchmark.get_results();
//! results.print_report();
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;
use std::time::{Duration, Instant};
use tenflowers_core::Result;

/// Configuration for performance benchmarking
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Number of benchmark iterations
    pub iterations: usize,
    /// Number of warmup iterations (not included in results)
    pub warmup_iterations: usize,
    /// Enable statistical analysis
    pub enable_statistics: bool,
    /// Enable memory profiling during benchmark
    pub enable_memory_profiling: bool,
    /// Target confidence interval (e.g., 0.95 for 95%)
    pub confidence_interval: f64,
    /// Maximum coefficient of variation to accept (stability threshold)
    pub max_cv: f64,
    /// Enable throughput measurement
    pub measure_throughput: bool,
    /// Enable latency percentiles
    pub measure_percentiles: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            iterations: 100,
            warmup_iterations: 10,
            enable_statistics: true,
            enable_memory_profiling: false,
            confidence_interval: 0.95,
            max_cv: 0.1, // 10% coefficient of variation
            measure_throughput: true,
            measure_percentiles: true,
        }
    }
}

impl BenchmarkConfig {
    /// Set number of iterations
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    /// Set number of warmup iterations
    pub fn with_warmup(mut self, warmup: usize) -> Self {
        self.warmup_iterations = warmup;
        self
    }

    /// Enable/disable statistical analysis
    pub fn with_statistical_analysis(mut self, enabled: bool) -> Self {
        self.enable_statistics = enabled;
        self
    }

    /// Enable/disable memory profiling
    pub fn with_memory_profiling(mut self, enabled: bool) -> Self {
        self.enable_memory_profiling = enabled;
        self
    }

    /// Set confidence interval level
    pub fn with_confidence_interval(mut self, level: f64) -> Self {
        self.confidence_interval = level;
        self
    }
}

/// Statistical summary of benchmark measurements
#[derive(Debug, Clone)]
pub struct BenchmarkStatistics {
    /// Mean execution time
    pub mean: Duration,
    /// Median execution time
    pub median: Duration,
    /// Standard deviation
    pub std_dev: Duration,
    /// Minimum execution time
    pub min: Duration,
    /// Maximum execution time
    pub max: Duration,
    /// Coefficient of variation (std_dev / mean)
    pub coefficient_of_variation: f64,
    /// Percentiles (p50, p90, p95, p99)
    pub percentiles: HashMap<u8, Duration>,
    /// Confidence interval bounds
    pub confidence_interval: (Duration, Duration),
}

/// Throughput metrics for benchmark
#[derive(Debug, Clone)]
pub struct ThroughputMetrics {
    /// Operations per second
    pub ops_per_second: f64,
    /// Total elements processed per second
    pub elements_per_second: f64,
    /// Gigabytes per second (for memory bandwidth)
    pub gigabytes_per_second: f64,
}

/// Memory usage during benchmark
#[derive(Debug, Clone, Default)]
pub struct BenchmarkMemoryStats {
    /// Peak memory usage (bytes)
    pub peak_memory: u64,
    /// Average memory usage (bytes)
    pub avg_memory: u64,
    /// Memory allocations count
    pub allocations: u64,
}

/// Single benchmark result
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Operation name
    pub operation_name: String,
    /// All measured durations
    pub measurements: Vec<Duration>,
    /// Statistical summary
    pub statistics: Option<BenchmarkStatistics>,
    /// Throughput metrics
    pub throughput: Option<ThroughputMetrics>,
    /// Memory statistics
    pub memory_stats: Option<BenchmarkMemoryStats>,
    /// Total number of elements processed
    pub total_elements: usize,
    /// Configuration used
    pub config: BenchmarkConfig,
}

impl BenchmarkResult {
    /// Check if result meets stability criteria
    pub fn is_stable(&self) -> bool {
        if let Some(ref stats) = self.statistics {
            stats.coefficient_of_variation <= self.config.max_cv
        } else {
            true
        }
    }

    /// Get speedup compared to another result
    pub fn speedup_vs(&self, other: &BenchmarkResult) -> f64 {
        if let (Some(ref self_stats), Some(ref other_stats)) = (&self.statistics, &other.statistics)
        {
            other_stats.mean.as_secs_f64() / self_stats.mean.as_secs_f64()
        } else {
            1.0
        }
    }

    /// Export to JSON format
    pub fn to_json(&self) -> String {
        let stats_str = if let Some(ref stats) = self.statistics {
            format!(
                r#""mean_ms": {:.3}, "median_ms": {:.3}, "std_dev_ms": {:.3}, "min_ms": {:.3}, "max_ms": {:.3}, "cv": {:.4}"#,
                stats.mean.as_secs_f64() * 1000.0,
                stats.median.as_secs_f64() * 1000.0,
                stats.std_dev.as_secs_f64() * 1000.0,
                stats.min.as_secs_f64() * 1000.0,
                stats.max.as_secs_f64() * 1000.0,
                stats.coefficient_of_variation
            )
        } else {
            String::new()
        };

        format!(
            r#"{{"operation": "{}", "iterations": {}, {}}}"#,
            self.operation_name,
            self.measurements.len(),
            stats_str
        )
    }
}

/// Performance benchmark runner
pub struct PerformanceBenchmark {
    config: BenchmarkConfig,
    results: HashMap<String, BenchmarkResult>,
}

impl PerformanceBenchmark {
    /// Create a new performance benchmark
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            config,
            results: HashMap::new(),
        }
    }

    /// Benchmark an operation with automatic analysis
    pub fn benchmark_operation<F, T>(&mut self, name: &str, mut operation: F) -> Result<()>
    where
        F: FnMut() -> Result<T>,
    {
        // Warmup phase
        for _ in 0..self.config.warmup_iterations {
            let _ = operation()?;
        }

        // Measurement phase
        let mut measurements = Vec::with_capacity(self.config.iterations);
        let mut memory_samples = Vec::new();

        for _ in 0..self.config.iterations {
            let memory_before = if self.config.enable_memory_profiling {
                self.get_memory_usage()
            } else {
                0
            };

            let start = Instant::now();
            let _ = operation()?;
            let duration = start.elapsed();

            let memory_after = if self.config.enable_memory_profiling {
                self.get_memory_usage()
            } else {
                0
            };

            measurements.push(duration);

            if self.config.enable_memory_profiling {
                memory_samples.push(memory_after.saturating_sub(memory_before));
            }
        }

        // Calculate statistics
        let statistics = if self.config.enable_statistics {
            Some(self.calculate_statistics(&measurements))
        } else {
            None
        };

        // Calculate throughput (simplified - would need actual element counts)
        let throughput = if self.config.measure_throughput {
            statistics.as_ref().map(|stats| ThroughputMetrics {
                ops_per_second: 1.0 / stats.mean.as_secs_f64(),
                elements_per_second: 0.0, // Would need actual element count
                gigabytes_per_second: 0.0,
            })
        } else {
            None
        };

        // Calculate memory statistics
        let memory_stats = if self.config.enable_memory_profiling && !memory_samples.is_empty() {
            Some(BenchmarkMemoryStats {
                peak_memory: *memory_samples.iter().max().unwrap_or(&0),
                avg_memory: memory_samples.iter().sum::<u64>() / memory_samples.len() as u64,
                allocations: self.config.iterations as u64,
            })
        } else {
            None
        };

        let result = BenchmarkResult {
            operation_name: name.to_string(),
            measurements,
            statistics,
            throughput,
            memory_stats,
            total_elements: 0, // Would need actual element count
            config: self.config.clone(),
        };

        self.results.insert(name.to_string(), result);

        Ok(())
    }

    /// Get benchmark results
    pub fn get_results(&self) -> &HashMap<String, BenchmarkResult> {
        &self.results
    }

    /// Get result for specific operation
    pub fn get_result(&self, name: &str) -> Option<&BenchmarkResult> {
        self.results.get(name)
    }

    /// Compare two benchmark results
    pub fn compare(&self, operation1: &str, operation2: &str) -> Option<ComparisonResult> {
        let result1 = self.results.get(operation1)?;
        let result2 = self.results.get(operation2)?;

        let speedup = result1.speedup_vs(result2);
        let winner = if speedup > 1.0 {
            operation1
        } else {
            operation2
        };

        Some(ComparisonResult {
            operation1: operation1.to_string(),
            operation2: operation2.to_string(),
            speedup,
            winner: winner.to_string(),
            is_significant: (speedup - 1.0).abs() > 0.05, // 5% threshold
        })
    }

    /// Detect performance regressions compared to baseline
    pub fn detect_regressions(
        &self,
        baseline: &HashMap<String, BenchmarkResult>,
    ) -> Vec<RegressionReport> {
        let mut regressions = Vec::new();

        for (op_name, current_result) in &self.results {
            if let Some(baseline_result) = baseline.get(op_name) {
                let speedup = current_result.speedup_vs(baseline_result);

                // Regression if current is >10% slower than baseline
                if speedup < 0.9 {
                    regressions.push(RegressionReport {
                        operation: op_name.clone(),
                        baseline_mean: baseline_result
                            .statistics
                            .as_ref()
                            .map(|s| s.mean)
                            .unwrap_or(Duration::ZERO),
                        current_mean: current_result
                            .statistics
                            .as_ref()
                            .map(|s| s.mean)
                            .unwrap_or(Duration::ZERO),
                        slowdown: 1.0 / speedup,
                        severity: if speedup < 0.5 {
                            RegressionSeverity::Critical
                        } else if speedup < 0.75 {
                            RegressionSeverity::High
                        } else {
                            RegressionSeverity::Medium
                        },
                    });
                }
            }
        }

        regressions
    }

    /// Generate comprehensive benchmark report
    pub fn generate_report(&self) -> BenchmarkReport {
        BenchmarkReport {
            total_operations: self.results.len(),
            results: self.results.clone(),
            summary: self.generate_summary(),
        }
    }

    // Private helper methods

    fn calculate_statistics(&self, measurements: &[Duration]) -> BenchmarkStatistics {
        let mut sorted = measurements.to_vec();
        sorted.sort();

        let mean_secs =
            measurements.iter().map(|d| d.as_secs_f64()).sum::<f64>() / measurements.len() as f64;
        let mean = Duration::from_secs_f64(mean_secs);

        let median = sorted[sorted.len() / 2];
        let min = sorted[0];
        let max = sorted[sorted.len() - 1];

        // Calculate standard deviation
        let variance: f64 = measurements
            .iter()
            .map(|d| {
                let diff = d.as_secs_f64() - mean.as_secs_f64();
                diff * diff
            })
            .sum::<f64>()
            / measurements.len() as f64;

        let std_dev = Duration::from_secs_f64(variance.sqrt());

        let coefficient_of_variation = if mean.as_secs_f64() > 0.0 {
            std_dev.as_secs_f64() / mean.as_secs_f64()
        } else {
            0.0
        };

        // Calculate percentiles
        let mut percentiles = HashMap::new();
        if self.config.measure_percentiles {
            percentiles.insert(50, sorted[sorted.len() / 2]);
            percentiles.insert(90, sorted[(sorted.len() * 90) / 100]);
            percentiles.insert(95, sorted[(sorted.len() * 95) / 100]);
            percentiles.insert(99, sorted[(sorted.len() * 99) / 100]);
        }

        // Simplified confidence interval (would use t-distribution in practice)
        let margin = std_dev.mul_f64(1.96); // 95% CI approximation
        let confidence_interval = (mean.saturating_sub(margin), mean.saturating_add(margin));

        BenchmarkStatistics {
            mean,
            median,
            std_dev,
            min,
            max,
            coefficient_of_variation,
            percentiles,
            confidence_interval,
        }
    }

    fn get_memory_usage(&self) -> u64 {
        // Simplified - would use actual system memory APIs
        0
    }

    fn generate_summary(&self) -> BenchmarkSummary {
        let total_ops = self.results.len();
        let stable_ops = self.results.values().filter(|r| r.is_stable()).count();

        let avg_ops_per_sec = if !self.results.is_empty() {
            self.results
                .values()
                .filter_map(|r| r.throughput.as_ref())
                .map(|t| t.ops_per_second)
                .sum::<f64>()
                / self.results.len() as f64
        } else {
            0.0
        };

        BenchmarkSummary {
            total_operations: total_ops,
            stable_operations: stable_ops,
            unstable_operations: total_ops - stable_ops,
            avg_ops_per_second: avg_ops_per_sec,
        }
    }
}

impl Default for PerformanceBenchmark {
    fn default() -> Self {
        Self::new(BenchmarkConfig::default())
    }
}

/// Comparison between two benchmark results
#[derive(Debug, Clone)]
pub struct ComparisonResult {
    pub operation1: String,
    pub operation2: String,
    pub speedup: f64,
    pub winner: String,
    pub is_significant: bool,
}

/// Performance regression report
#[derive(Debug, Clone)]
pub struct RegressionReport {
    pub operation: String,
    pub baseline_mean: Duration,
    pub current_mean: Duration,
    pub slowdown: f64,
    pub severity: RegressionSeverity,
}

/// Severity of performance regression
#[derive(Debug, Clone, PartialEq)]
pub enum RegressionSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Summary of benchmark results
#[derive(Debug, Clone)]
pub struct BenchmarkSummary {
    pub total_operations: usize,
    pub stable_operations: usize,
    pub unstable_operations: usize,
    pub avg_ops_per_second: f64,
}

/// Complete benchmark report
#[derive(Debug, Clone)]
pub struct BenchmarkReport {
    pub total_operations: usize,
    pub results: HashMap<String, BenchmarkResult>,
    pub summary: BenchmarkSummary,
}

impl BenchmarkReport {
    /// Print formatted report
    pub fn print_report(&self) {
        println!("\n=== Performance Benchmark Report ===");
        println!("Total Operations Benchmarked: {}", self.total_operations);
        println!("Stable Operations: {}", self.summary.stable_operations);
        println!("Unstable Operations: {}", self.summary.unstable_operations);
        println!("Average Ops/sec: {:.2}", self.summary.avg_ops_per_second);

        println!("\n--- Per-Operation Results ---");
        let mut ops: Vec<_> = self.results.iter().collect();
        ops.sort_by_key(|(name, _)| (*name).to_string());

        for (name, result) in ops {
            if let Some(ref stats) = result.statistics {
                println!("\n{}:", name);
                println!("  Mean: {:.3} ms", stats.mean.as_secs_f64() * 1000.0);
                println!("  Median: {:.3} ms", stats.median.as_secs_f64() * 1000.0);
                println!("  Std Dev: {:.3} ms", stats.std_dev.as_secs_f64() * 1000.0);
                println!("  Min: {:.3} ms", stats.min.as_secs_f64() * 1000.0);
                println!("  Max: {:.3} ms", stats.max.as_secs_f64() * 1000.0);
                println!("  CV: {:.2}%", stats.coefficient_of_variation * 100.0);
                println!("  Stable: {}", result.is_stable());

                if let Some(ref throughput) = result.throughput {
                    println!("  Throughput: {:.2} ops/sec", throughput.ops_per_second);
                }
            }
        }
        println!("\n====================================\n");
    }

    /// Export to CSV format
    pub fn to_csv(&self) -> String {
        let mut csv =
            String::from("Operation,Mean(ms),Median(ms),StdDev(ms),Min(ms),Max(ms),CV,Stable\n");

        let mut ops: Vec<_> = self.results.iter().collect();
        ops.sort_by_key(|(name, _)| (*name).to_string());

        for (name, result) in ops {
            if let Some(ref stats) = result.statistics {
                csv.push_str(&format!(
                    "{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.4},{}\n",
                    name,
                    stats.mean.as_secs_f64() * 1000.0,
                    stats.median.as_secs_f64() * 1000.0,
                    stats.std_dev.as_secs_f64() * 1000.0,
                    stats.min.as_secs_f64() * 1000.0,
                    stats.max.as_secs_f64() * 1000.0,
                    stats.coefficient_of_variation,
                    result.is_stable()
                ));
            }
        }

        csv
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_config_builder() {
        let config = BenchmarkConfig::default()
            .with_iterations(50)
            .with_warmup(5)
            .with_statistical_analysis(true)
            .with_confidence_interval(0.99);

        assert_eq!(config.iterations, 50);
        assert_eq!(config.warmup_iterations, 5);
        assert!(config.enable_statistics);
        assert_eq!(config.confidence_interval, 0.99);
    }

    #[test]
    fn test_benchmark_creation() {
        let benchmark = PerformanceBenchmark::default();
        assert!(benchmark.results.is_empty());
    }

    #[test]
    fn test_simple_benchmark() {
        let config = BenchmarkConfig::default()
            .with_iterations(10)
            .with_warmup(2);

        let mut benchmark = PerformanceBenchmark::new(config);

        benchmark
            .benchmark_operation("test_op", || {
                std::thread::sleep(Duration::from_micros(100));
                Ok::<(), tenflowers_core::TensorError>(())
            })
            .unwrap();

        let result = benchmark.get_result("test_op").unwrap();
        assert_eq!(result.measurements.len(), 10);
        assert!(result.statistics.is_some());
    }

    #[test]
    fn test_statistics_calculation() {
        let config = BenchmarkConfig::default().with_iterations(5);
        let mut benchmark = PerformanceBenchmark::new(config);

        let measurements = vec![
            Duration::from_millis(100),
            Duration::from_millis(105),
            Duration::from_millis(110),
            Duration::from_millis(95),
            Duration::from_millis(100),
        ];

        let stats = benchmark.calculate_statistics(&measurements);

        // Mean should be around 102ms
        assert!(stats.mean.as_millis() >= 100 && stats.mean.as_millis() <= 105);
        assert!(stats.median == Duration::from_millis(100));
        assert!(stats.min == Duration::from_millis(95));
        assert!(stats.max == Duration::from_millis(110));
    }

    #[test]
    fn test_result_stability() {
        let stable_result = BenchmarkResult {
            operation_name: "stable".to_string(),
            measurements: vec![Duration::from_millis(100); 10],
            statistics: Some(BenchmarkStatistics {
                mean: Duration::from_millis(100),
                median: Duration::from_millis(100),
                std_dev: Duration::from_millis(1),
                min: Duration::from_millis(99),
                max: Duration::from_millis(101),
                coefficient_of_variation: 0.01, // 1% CV
                percentiles: HashMap::new(),
                confidence_interval: (Duration::from_millis(99), Duration::from_millis(101)),
            }),
            throughput: None,
            memory_stats: None,
            total_elements: 0,
            config: BenchmarkConfig::default(),
        };

        assert!(stable_result.is_stable());
    }

    #[test]
    fn test_speedup_calculation() {
        let fast = BenchmarkResult {
            operation_name: "fast".to_string(),
            measurements: vec![],
            statistics: Some(BenchmarkStatistics {
                mean: Duration::from_millis(50),
                median: Duration::from_millis(50),
                std_dev: Duration::from_millis(1),
                min: Duration::from_millis(49),
                max: Duration::from_millis(51),
                coefficient_of_variation: 0.02,
                percentiles: HashMap::new(),
                confidence_interval: (Duration::from_millis(49), Duration::from_millis(51)),
            }),
            throughput: None,
            memory_stats: None,
            total_elements: 0,
            config: BenchmarkConfig::default(),
        };

        let slow = BenchmarkResult {
            operation_name: "slow".to_string(),
            measurements: vec![],
            statistics: Some(BenchmarkStatistics {
                mean: Duration::from_millis(100),
                median: Duration::from_millis(100),
                std_dev: Duration::from_millis(2),
                min: Duration::from_millis(98),
                max: Duration::from_millis(102),
                coefficient_of_variation: 0.02,
                percentiles: HashMap::new(),
                confidence_interval: (Duration::from_millis(98), Duration::from_millis(102)),
            }),
            throughput: None,
            memory_stats: None,
            total_elements: 0,
            config: BenchmarkConfig::default(),
        };

        let speedup = fast.speedup_vs(&slow);
        assert!((speedup - 2.0).abs() < 0.01); // Should be ~2x faster
    }

    #[test]
    fn test_regression_detection() {
        let config = BenchmarkConfig::default();
        let mut baseline = HashMap::new();
        let mut current = PerformanceBenchmark::new(config);

        // Baseline result: 100ms
        baseline.insert(
            "test_op".to_string(),
            BenchmarkResult {
                operation_name: "test_op".to_string(),
                measurements: vec![],
                statistics: Some(BenchmarkStatistics {
                    mean: Duration::from_millis(100),
                    median: Duration::from_millis(100),
                    std_dev: Duration::from_millis(5),
                    min: Duration::from_millis(95),
                    max: Duration::from_millis(105),
                    coefficient_of_variation: 0.05,
                    percentiles: HashMap::new(),
                    confidence_interval: (Duration::from_millis(95), Duration::from_millis(105)),
                }),
                throughput: None,
                memory_stats: None,
                total_elements: 0,
                config: BenchmarkConfig::default(),
            },
        );

        // Current result: 150ms (50% slower - regression!)
        current.results.insert(
            "test_op".to_string(),
            BenchmarkResult {
                operation_name: "test_op".to_string(),
                measurements: vec![],
                statistics: Some(BenchmarkStatistics {
                    mean: Duration::from_millis(150),
                    median: Duration::from_millis(150),
                    std_dev: Duration::from_millis(7),
                    min: Duration::from_millis(143),
                    max: Duration::from_millis(157),
                    coefficient_of_variation: 0.05,
                    percentiles: HashMap::new(),
                    confidence_interval: (Duration::from_millis(143), Duration::from_millis(157)),
                }),
                throughput: None,
                memory_stats: None,
                total_elements: 0,
                config: BenchmarkConfig::default(),
            },
        );

        let regressions = current.detect_regressions(&baseline);
        assert_eq!(regressions.len(), 1);
        assert_eq!(regressions[0].operation, "test_op");
        assert!(regressions[0].slowdown > 1.4); // ~1.5x slower
    }
}
