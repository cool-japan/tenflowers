//! Performance metrics and adaptive threshold types for kernel fusion.

use std::collections::HashMap;
use std::time::Instant;

/// Historical performance measurement
#[derive(Debug, Clone)]
pub struct PerformanceDataPoint {
    /// Execution time in microseconds
    pub execution_time: f64,
    /// Memory bandwidth utilization
    pub bandwidth_utilization: f64,
    /// Compute utilization percentage
    pub compute_utilization: f64,
    /// Timestamp of measurement
    pub timestamp: Instant,
    /// Input shape that produced this measurement
    pub input_shape: Vec<usize>,
}

/// Performance profiling for ML-based optimization
#[derive(Debug, Clone)]
pub struct PerformanceProfile {
    /// Estimated FLOPs for the fused operation
    pub estimated_flops: u64,
    /// Memory bandwidth requirements (bytes/second)
    pub memory_bandwidth: u64,
    /// Arithmetic intensity (FLOPs per byte)
    pub arithmetic_intensity: f64,
    /// Estimated execution time (microseconds)
    pub estimated_latency: f64,
    /// Cache efficiency score (0.0 to 1.0)
    pub cache_efficiency: f64,
    /// Parallel efficiency potential (0.0 to 1.0)
    pub parallel_efficiency: f64,
    /// Historical performance data
    pub historical_performance: Vec<PerformanceDataPoint>,
}

/// Sophisticated adaptive thresholds for fusion decisions
#[derive(Debug, Clone)]
pub struct AdaptiveThresholds {
    pub min_fusion_benefit: f32,
    pub max_compilation_time_ms: f64,
    pub memory_pressure_threshold: f32,
    pub thermal_throttling_threshold: f32,
}

/// Sophisticated performance metrics for fusion analytics
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub execution_time_ms: f64,
    pub memory_bandwidth_gbps: f64,
    pub compute_throughput_tflops: f64,
    pub cache_hit_ratio: f64,
    pub energy_efficiency: f64,
    pub fusion_effectiveness: f64,
}

/// Ultra-sophisticated optimization levels
#[derive(Debug, Clone, Copy)]
pub enum OptimizationLevel {
    Conservative,
    Moderate,
    Aggressive,
    UltraOptimized,
    ProductionMaximized,
}

/// Ultra-sophisticated adaptive fusion strategy
#[derive(Debug, Clone)]
pub struct AdaptiveFusionStrategy {
    pub learning_rate: f32,
    pub performance_history: Vec<PerformanceMetrics>,
    pub optimization_decisions: HashMap<String, OptimizationLevel>,
    pub adaptive_thresholds: AdaptiveThresholds,
}
