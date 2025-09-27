//! 🚀 Ultra-Performance Profiler Integration
//!
//! This module provides comprehensive performance monitoring and profiling
//! capabilities for TenflowRS, integrating with the SciRS2 metrics system
//! and providing real-time performance analytics.

// TODO: Add back when SciRS2 metrics integration is implemented
// use scirs2_core::metrics::{Counter, Timer};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Global performance profiler instance
static PROFILER: once_cell::sync::Lazy<Arc<Mutex<UltraPerformanceProfiler>>> =
    once_cell::sync::Lazy::new(|| Arc::new(Mutex::new(UltraPerformanceProfiler::new())));

/// Ultra-performance profiler for comprehensive monitoring
pub struct UltraPerformanceProfiler {
    /// Operation performance metrics
    operation_metrics: HashMap<String, OperationMetrics>,
    /// Real-time performance history
    performance_history: Vec<PerformanceDataPoint>,
    /// Configuration settings
    config: ProfilerConfig,
    /// Global performance counters
    counters: PerformanceCounters,
}

/// Performance metrics for a specific operation
#[derive(Debug, Clone)]
pub struct OperationMetrics {
    /// Total number of calls
    pub call_count: u64,
    /// Total execution time
    pub total_time: Duration,
    /// Minimum execution time
    pub min_time: Duration,
    /// Maximum execution time
    pub max_time: Duration,
    /// Average execution time
    pub avg_time: Duration,
    /// Total FLOPs processed
    pub total_flops: u64,
    /// Peak GFLOP/s achieved
    pub peak_gflops: f64,
    /// Memory throughput statistics
    pub memory_stats: MemoryStats,
}

/// Memory usage statistics
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    /// Total bytes processed
    pub total_bytes: u64,
    /// Peak memory bandwidth (GB/s)
    pub peak_bandwidth: f64,
    /// Cache utilization efficiency
    pub cache_efficiency: f64,
}

/// Performance data point for historical tracking
#[derive(Debug, Clone)]
pub struct PerformanceDataPoint {
    /// Timestamp of measurement
    pub timestamp: Instant,
    /// Operation name
    pub operation: String,
    /// Matrix dimensions (m, n, k) for matmul operations
    pub dimensions: (usize, usize, usize),
    /// Execution time
    pub execution_time: Duration,
    /// Performance in GFLOP/s
    pub gflops: f64,
    /// Memory bandwidth in GB/s
    pub bandwidth: f64,
}

/// Profiler configuration
#[derive(Debug, Clone)]
pub struct ProfilerConfig {
    /// Enable detailed profiling
    pub detailed_profiling: bool,
    /// Maximum history entries to keep
    pub max_history_entries: usize,
    /// Minimum operation time to record (nanoseconds)
    pub min_record_time: u64,
    /// Enable real-time optimization recommendations
    pub optimization_recommendations: bool,
}

/// Global performance counters
#[derive(Debug, Clone, Default)]
pub struct PerformanceCounters {
    /// Total matrix operations performed
    pub total_matmul_ops: u64,
    /// Total FLOPs processed
    pub total_flops: u64,
    /// Total computation time
    pub total_compute_time: Duration,
    /// Peak performance achieved
    pub peak_gflops: f64,
}

impl Default for UltraPerformanceProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl UltraPerformanceProfiler {
    /// Create a new ultra-performance profiler
    pub fn new() -> Self {
        Self {
            operation_metrics: HashMap::new(),
            performance_history: Vec::new(),
            config: ProfilerConfig::default(),
            counters: PerformanceCounters::default(),
        }
    }

    /// Record a matrix multiplication operation
    pub fn record_matmul(
        &mut self,
        operation: &str,
        m: usize,
        n: usize,
        k: usize,
        elapsed: Duration,
    ) {
        // Calculate performance metrics
        let flops = 2 * m * n * k; // FMAOPs for matrix multiplication
        let gflops = flops as f64 / elapsed.as_secs_f64() / 1e9;

        // Estimate memory usage (read A, read B, write C)
        let bytes = (m * k + k * n + m * n) * 4; // Assuming f32
        let bandwidth = bytes as f64 / elapsed.as_secs_f64() / 1e9; // GB/s

        // Update operation metrics
        let metrics = self
            .operation_metrics
            .entry(operation.to_string())
            .or_insert_with(|| OperationMetrics {
                call_count: 0,
                total_time: Duration::ZERO,
                min_time: Duration::MAX,
                max_time: Duration::ZERO,
                avg_time: Duration::ZERO,
                total_flops: 0,
                peak_gflops: 0.0,
                memory_stats: MemoryStats::default(),
            });

        // Update metrics
        metrics.call_count += 1;
        metrics.total_time += elapsed;
        metrics.min_time = metrics.min_time.min(elapsed);
        metrics.max_time = metrics.max_time.max(elapsed);
        metrics.avg_time = metrics.total_time / metrics.call_count as u32;
        metrics.total_flops += flops as u64;
        metrics.peak_gflops = metrics.peak_gflops.max(gflops);
        metrics.memory_stats.total_bytes += bytes as u64;
        metrics.memory_stats.peak_bandwidth = metrics.memory_stats.peak_bandwidth.max(bandwidth);

        // Update global counters
        self.counters.total_matmul_ops += 1;
        self.counters.total_flops += flops as u64;
        self.counters.total_compute_time += elapsed;
        self.counters.peak_gflops = self.counters.peak_gflops.max(gflops);

        // Add to performance history
        if self.config.detailed_profiling
            && elapsed.as_nanos() as u64 >= self.config.min_record_time
        {
            self.performance_history.push(PerformanceDataPoint {
                timestamp: Instant::now(),
                operation: operation.to_string(),
                dimensions: (m, n, k),
                execution_time: elapsed,
                gflops,
                bandwidth,
            });

            // Limit history size
            if self.performance_history.len() > self.config.max_history_entries {
                self.performance_history.remove(0);
            }
        }

        // Log significant performance achievements
        if gflops > 100.0 {
            println!(
                "🚀 HIGH PERFORMANCE: {} achieved {:.2} GFLOP/s on {}x{}x{}",
                operation, gflops, m, n, k
            );
        }
    }

    /// Get performance summary
    pub fn get_performance_summary(&self) -> PerformanceSummary {
        let total_ops = self.counters.total_matmul_ops;
        let avg_gflops = if total_ops > 0 {
            self.counters.total_flops as f64 / self.counters.total_compute_time.as_secs_f64() / 1e9
        } else {
            0.0
        };

        PerformanceSummary {
            total_operations: total_ops,
            total_flops: self.counters.total_flops,
            total_compute_time: self.counters.total_compute_time,
            average_gflops: avg_gflops,
            peak_gflops: self.counters.peak_gflops,
            operation_count: self.operation_metrics.len(),
            recent_performance: self.get_recent_performance_trend(),
        }
    }

    /// Get recent performance trend
    fn get_recent_performance_trend(&self) -> f64 {
        const RECENT_WINDOW: usize = 10;
        let recent_count = self.performance_history.len().min(RECENT_WINDOW);

        if recent_count < 2 {
            return 0.0;
        }

        let recent_entries =
            &self.performance_history[self.performance_history.len() - recent_count..];
        let avg_recent_gflops: f64 =
            recent_entries.iter().map(|p| p.gflops).sum::<f64>() / recent_count as f64;

        avg_recent_gflops
    }

    /// Print detailed performance report
    pub fn print_performance_report(&self) {
        println!("\n🚀 ULTRA-PERFORMANCE PROFILER REPORT");
        println!("{}", "=".repeat(60));

        let summary = self.get_performance_summary();
        println!("📊 OVERALL PERFORMANCE:");
        println!("   Total Operations:     {}", summary.total_operations);
        println!(
            "   Total FLOPs:          {:.2e}",
            summary.total_flops as f64
        );
        println!(
            "   Total Compute Time:   {:.2}ms",
            summary.total_compute_time.as_secs_f64() * 1000.0
        );
        println!(
            "   Average Performance:  {:.2} GFLOP/s",
            summary.average_gflops
        );
        println!(
            "   Peak Performance:     {:.2} GFLOP/s",
            summary.peak_gflops
        );
        println!(
            "   Recent Trend:         {:.2} GFLOP/s",
            summary.recent_performance
        );
        println!();

        println!("📋 OPERATION BREAKDOWN:");
        for (op_name, metrics) in &self.operation_metrics {
            println!("   {} ({} calls):", op_name, metrics.call_count);
            println!(
                "     Avg Time:     {:.2}ms",
                metrics.avg_time.as_secs_f64() * 1000.0
            );
            println!(
                "     Min Time:     {:.2}ms",
                metrics.min_time.as_secs_f64() * 1000.0
            );
            println!(
                "     Max Time:     {:.2}ms",
                metrics.max_time.as_secs_f64() * 1000.0
            );
            println!("     Peak GFLOP/s: {:.2}", metrics.peak_gflops);
            println!(
                "     Memory B/W:   {:.2} GB/s",
                metrics.memory_stats.peak_bandwidth
            );
            println!();
        }

        if self.config.optimization_recommendations {
            self.print_optimization_recommendations();
        }
    }

    /// Print optimization recommendations
    fn print_optimization_recommendations(&self) {
        println!("💡 OPTIMIZATION RECOMMENDATIONS:");

        for (op_name, metrics) in &self.operation_metrics {
            if metrics.peak_gflops < 50.0 && metrics.call_count > 10 {
                println!(
                    "   ⚡ {}: Consider SIMD optimization (current: {:.2} GFLOP/s)",
                    op_name, metrics.peak_gflops
                );
            }

            if metrics.memory_stats.peak_bandwidth < 10.0 && metrics.call_count > 5 {
                println!(
                    "   🧠 {}: Memory bandwidth limited (current: {:.2} GB/s)",
                    op_name, metrics.memory_stats.peak_bandwidth
                );
            }
        }

        if self.counters.peak_gflops > 500.0 {
            println!("   🏆 EXCELLENT: Peak performance exceeds 500 GFLOP/s!");
        }

        println!();
    }

    /// Configure profiler settings
    pub fn configure(&mut self, config: ProfilerConfig) {
        self.config = config;
    }

    /// Clear all performance data
    pub fn clear(&mut self) {
        self.operation_metrics.clear();
        self.performance_history.clear();
        self.counters = PerformanceCounters::default();
    }
}

/// Performance summary structure
#[derive(Debug, Clone)]
pub struct PerformanceSummary {
    pub total_operations: u64,
    pub total_flops: u64,
    pub total_compute_time: Duration,
    pub average_gflops: f64,
    pub peak_gflops: f64,
    pub operation_count: usize,
    pub recent_performance: f64,
}

impl Default for ProfilerConfig {
    fn default() -> Self {
        Self {
            detailed_profiling: true,
            max_history_entries: 1000,
            min_record_time: 1000, // 1 microsecond
            optimization_recommendations: true,
        }
    }
}

/// Global profiler access functions
pub fn record_matmul_performance(operation: &str, m: usize, n: usize, k: usize, elapsed: Duration) {
    if let Ok(mut profiler) = PROFILER.lock() {
        profiler.record_matmul(operation, m, n, k, elapsed);
    }
}

pub fn get_performance_summary() -> Option<PerformanceSummary> {
    PROFILER
        .lock()
        .ok()
        .map(|profiler| profiler.get_performance_summary())
}

pub fn print_performance_report() {
    if let Ok(profiler) = PROFILER.lock() {
        profiler.print_performance_report();
    }
}

pub fn configure_profiler(config: ProfilerConfig) {
    if let Ok(mut profiler) = PROFILER.lock() {
        profiler.configure(config);
    }
}

pub fn clear_performance_data() {
    if let Ok(mut profiler) = PROFILER.lock() {
        profiler.clear();
    }
}

/// Performance measurement macro
#[macro_export]
macro_rules! measure_performance {
    ($operation:expr, $m:expr, $n:expr, $k:expr, $block:block) => {{
        let start = std::time::Instant::now();
        let result = $block;
        let elapsed = start.elapsed();
        $crate::ultra_performance_profiler::record_matmul_performance(
            $operation, $m, $n, $k, elapsed,
        );
        result
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_basic_functionality() {
        let mut profiler = UltraPerformanceProfiler::new();

        // Record some operations
        profiler.record_matmul("test_matmul", 64, 64, 64, Duration::from_millis(1));
        profiler.record_matmul("test_matmul", 128, 128, 128, Duration::from_millis(5));

        let summary = profiler.get_performance_summary();
        assert_eq!(summary.total_operations, 2);
        assert!(summary.average_gflops > 0.0);
        assert!(summary.peak_gflops > 0.0);
    }

    #[test]
    fn test_profiler_configuration() {
        let mut profiler = UltraPerformanceProfiler::new();

        let config = ProfilerConfig {
            detailed_profiling: false,
            max_history_entries: 100,
            min_record_time: 10000,
            optimization_recommendations: false,
        };

        profiler.configure(config.clone());
        assert_eq!(profiler.config.max_history_entries, 100);
        assert!(!profiler.config.detailed_profiling);
    }

    #[test]
    fn test_performance_metrics_calculation() {
        let mut profiler = UltraPerformanceProfiler::new();

        // Record a known operation
        let m = 100;
        let n = 100;
        let k = 100;
        let elapsed = Duration::from_millis(1);

        profiler.record_matmul("test_op", m, n, k, elapsed);

        let metrics = profiler.operation_metrics.get("test_op").unwrap();
        assert_eq!(metrics.call_count, 1);
        assert_eq!(metrics.total_time, elapsed);
        assert!(metrics.peak_gflops > 0.0);
    }
}
