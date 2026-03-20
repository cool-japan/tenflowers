//! Performance metrics and statistics for gradient computation

use std::time::Duration;

/// Ultra-fast gradient computation result
#[derive(Debug, Clone)]
pub struct UltraGradientResult<T> {
    /// Computed gradients with maximum efficiency
    pub gradients: std::collections::HashMap<u64, tenflowers_core::Tensor<T>>,
    /// Detailed performance metrics
    pub metrics: GradientPerformanceMetrics,
    /// Memory usage statistics
    pub memory_stats: GradientMemoryStats,
    /// Optimization insights
    pub insights: OptimizationInsights,
}

/// Comprehensive performance metrics for gradient computation
#[derive(Debug, Clone, Default)]
pub struct GradientPerformanceMetrics {
    /// Total computation time
    pub total_time: Duration,
    /// Time spent on gradient computation
    pub gradient_time: Duration,
    /// Time spent on memory operations
    pub memory_time: Duration,
    /// Time spent on SIMD operations
    pub simd_time: Duration,
    /// Number of operations processed
    pub operations_count: u64,
    /// Number of cache hits
    pub cache_hits: u64,
    /// Number of cache misses
    pub cache_misses: u64,
    /// Parallelization efficiency
    pub parallelization_efficiency: f64,
    /// SIMD utilization percentage
    pub simd_utilization: f64,
    /// Memory bandwidth utilization
    pub memory_bandwidth_utilization: f64,
}

/// Memory usage statistics for gradient computation
#[derive(Debug, Clone, Default)]
pub struct GradientMemoryStats {
    /// Peak memory usage during computation
    pub peak_memory_usage: u64,
    /// Total memory allocated
    pub total_memory_allocated: u64,
    /// Memory reused from buffer pool
    pub memory_reused: u64,
    /// Memory fragmentation ratio
    pub fragmentation_ratio: f64,
    /// Buffer pool efficiency
    pub buffer_pool_efficiency: f64,
}

/// Optimization insights for continuous performance improvement
#[derive(Debug, Clone, Default)]
pub struct OptimizationInsights {
    /// Recommended optimizations
    pub recommendations: Vec<String>,
    /// Performance bottlenecks identified
    pub bottlenecks: Vec<String>,
    /// Potential speedup estimates
    pub speedup_estimates: Vec<(String, f64)>,
    /// Memory optimization opportunities
    pub memory_optimizations: Vec<String>,
}
