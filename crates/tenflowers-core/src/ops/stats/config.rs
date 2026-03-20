//! Configuration, metrics structures and global state for statistical operations.

use scirs2_core::metrics::MetricsRegistry;
use std::sync::RwLock;
use std::time::Duration;

#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

/// Ultra-Performance Statistical Operations Configuration
///
/// Comprehensive configuration for statistical operations with advanced optimization
/// strategies including SIMD acceleration, parallel processing, and adaptive tuning.
#[derive(Debug, Clone)]
pub struct StatsConfig {
    /// Enable SIMD acceleration for statistical computations
    pub enable_simd: bool,
    /// Minimum array size for SIMD activation
    pub simd_threshold: usize,
    /// Enable parallel processing for large datasets
    pub enable_parallel: bool,
    /// Minimum array size for parallel processing
    pub parallel_threshold: usize,
    /// Number of worker threads (0 = auto-detect)
    pub num_threads: usize,
    /// Enable memory-efficient algorithms for large arrays
    pub enable_memory_efficient: bool,
    /// Memory mapping threshold for large datasets
    pub memory_mapping_threshold: usize,
    /// Enable adaptive chunking for progressive computation
    pub enable_adaptive_chunking: bool,
    /// Chunk size for adaptive processing
    pub adaptive_chunk_size: usize,
    /// Enable numerical stability optimizations
    pub enable_numerical_stability: bool,
    /// Enable performance monitoring
    pub enable_performance_monitoring: bool,
    /// Enable cache-friendly memory access optimization
    pub enable_cache_optimization: bool,
    /// Target numerical precision
    pub target_precision: f64,
}

impl Default for StatsConfig {
    fn default() -> Self {
        Self {
            enable_simd: true,
            simd_threshold: 1024,
            enable_parallel: true,
            parallel_threshold: 100_000,
            num_threads: 0, // Auto-detect
            enable_memory_efficient: true,
            memory_mapping_threshold: 100 * 1024 * 1024, // 100MB
            enable_adaptive_chunking: true,
            adaptive_chunk_size: 64 * 1024, // 64KB chunks
            enable_numerical_stability: true,
            enable_performance_monitoring: true,
            enable_cache_optimization: true,
            target_precision: 1e-12,
        }
    }
}

/// Advanced Statistical Metrics with Performance Analytics
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct StatisticalMetrics {
    /// Operation name
    pub operation: String,
    /// Input tensor size
    pub input_size: usize,
    /// Computation time
    pub computation_time: Duration,
    /// Memory usage (bytes)
    pub memory_usage: usize,
    /// SIMD utilization percentage
    pub simd_utilization: f64,
    /// Parallel efficiency (0.0 - 1.0)
    pub parallel_efficiency: f64,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Numerical error estimate
    pub numerical_error: f64,
    /// Performance score (operations per second)
    pub performance_score: f64,
}

// Global statistics configuration and performance metrics
lazy_static::lazy_static! {
    pub(super) static ref STATS_CONFIG: RwLock<StatsConfig> = RwLock::new(StatsConfig::default());
    pub(super) static ref PERFORMANCE_METRICS: RwLock<Vec<StatisticalMetrics>> = RwLock::new(Vec::new());
    pub(super) static ref METRICS_REGISTRY: MetricsRegistry = MetricsRegistry::new();
}

/// Get current statistical operations configuration
pub fn get_stats_config() -> StatsConfig {
    STATS_CONFIG
        .read()
        .expect("read lock should not be poisoned")
        .clone()
}

/// Update statistical operations configuration
pub fn set_stats_config(config: StatsConfig) {
    if let Ok(mut global_config) = STATS_CONFIG.write() {
        *global_config = config;
    }
}

/// Get performance metrics for statistical operations
pub fn get_performance_metrics() -> Vec<StatisticalMetrics> {
    PERFORMANCE_METRICS
        .read()
        .expect("read lock should not be poisoned")
        .clone()
}

/// Clear performance metrics history
pub fn clear_performance_metrics() {
    if let Ok(mut metrics) = PERFORMANCE_METRICS.write() {
        metrics.clear();
    }
}

/// Generate performance report for statistical operations
pub fn generate_performance_report() -> String {
    let metrics = get_performance_metrics();
    if metrics.is_empty() {
        return "No performance metrics available".to_string();
    }

    let total_ops = metrics.len();
    let avg_time = metrics
        .iter()
        .map(|m| m.computation_time.as_secs_f64())
        .sum::<f64>()
        / total_ops as f64;
    let avg_perf_score =
        metrics.iter().map(|m| m.performance_score).sum::<f64>() / total_ops as f64;

    format!(
        "Statistical Operations Performance Report\n\
         ==========================================\n\
         Total Operations: {}\n\
         Average Computation Time: {:.3}ms\n\
         Average Performance Score: {:.2} ops/sec\n\
         SIMD Utilization: {:.1}%\n\
         Parallel Efficiency: {:.1}%\n",
        total_ops,
        avg_time * 1000.0,
        avg_perf_score,
        metrics.iter().map(|m| m.simd_utilization).sum::<f64>() / total_ops as f64 * 100.0,
        metrics.iter().map(|m| m.parallel_efficiency).sum::<f64>() / total_ops as f64 * 100.0,
    )
}

/// Record statistical operation performance metrics
pub(super) fn record_stats_metrics(
    operation: &str,
    input_size: usize,
    computation_time: Duration,
    simd_utilization: f64,
    parallel_efficiency: f64,
) {
    let metrics = StatisticalMetrics {
        operation: operation.to_string(),
        input_size,
        computation_time,
        memory_usage: input_size * std::mem::size_of::<f64>(), // Estimate
        simd_utilization,
        parallel_efficiency,
        cache_hit_rate: 0.0,  // Would be measured in real implementation
        numerical_error: 0.0, // Would be estimated based on algorithm
        performance_score: input_size as f64 / computation_time.as_secs_f64(),
    };

    if let Ok(mut metrics_vec) = PERFORMANCE_METRICS.write() {
        metrics_vec.push(metrics);
        // Keep only recent metrics to prevent unbounded growth
        if metrics_vec.len() > 1000 {
            metrics_vec.drain(0..500);
        }
    }
}
