//! Performance metrics data structures
//!
//! This module contains all the data structures used for collecting, storing,
//! and managing performance metrics in the ultra performance monitoring system.

#![allow(dead_code)]

use crate::memory::UnifiedOptimizationStatistics;
use crate::simd::ultra_simd_engine::SimdPerformanceStats;
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, SystemTime};

/// Real-time metrics collection system
#[allow(dead_code)]
pub struct MetricsCollector {
    /// Performance metrics history
    pub(crate) metrics_history: VecDeque<PerformanceSnapshot>,
    /// Current system metrics
    pub(crate) current_metrics: SystemMetrics,
    /// Operation-specific metrics
    pub(crate) operation_metrics: HashMap<String, OperationMetrics>,
    /// Resource utilization metrics
    pub(crate) resource_metrics: ResourceMetrics,
    /// Custom metrics registry
    pub(crate) custom_metrics: HashMap<String, CustomMetric>,
}

/// Complete performance snapshot at a point in time
#[derive(Debug, Clone)]
pub struct PerformanceSnapshot {
    /// Timestamp
    pub timestamp: SystemTime,
    /// System-wide metrics
    pub system_metrics: SystemMetrics,
    /// SIMD performance metrics
    pub simd_metrics: Option<SimdPerformanceStats>,
    /// Memory optimization metrics
    pub memory_metrics: Option<UnifiedOptimizationStatistics>,
    /// Operation performance metrics
    pub operation_metrics: HashMap<String, OperationMetrics>,
    /// Resource utilization
    pub resource_utilization: ResourceMetrics,
    /// Performance quality score
    pub quality_score: f64,
}

/// System-wide performance metrics
#[derive(Debug, Clone)]
pub struct SystemMetrics {
    /// Total operations per second
    pub total_ops_per_second: f64,
    /// Average latency (milliseconds)
    pub average_latency_ms: f64,
    /// Memory usage (bytes)
    pub memory_usage: u64,
    /// CPU utilization (0-1)
    pub cpu_utilization: f64,
    /// Cache hit rate (0-1)
    pub cache_hit_rate: f64,
    /// Network bandwidth utilization (bytes/sec)
    pub network_bandwidth: u64,
    /// Disk I/O rate (bytes/sec)
    pub disk_io_rate: u64,
    /// Error rate (0-1)
    pub error_rate: f64,
}

/// Operation-specific performance metrics
#[derive(Debug, Clone)]
pub struct OperationMetrics {
    /// Operation name
    pub operation_name: String,
    /// Execution count
    pub execution_count: u64,
    /// Average execution time
    pub average_execution_time: Duration,
    /// Peak execution time
    pub peak_execution_time: Duration,
    /// Minimum execution time
    pub min_execution_time: Duration,
    /// Throughput (ops/sec)
    pub throughput: f64,
    /// Success rate (0-1)
    pub success_rate: f64,
    /// Memory usage per operation
    pub memory_per_operation: u64,
    /// Optimization effectiveness
    pub optimization_effectiveness: f64,
}

/// Resource utilization metrics
#[derive(Debug, Clone, Default)]
pub struct ResourceMetrics {
    /// CPU cores utilization
    pub cpu_cores: Vec<f64>,
    /// Memory segments utilization
    pub memory_segments: MemorySegmentMetrics,
    /// Cache levels utilization
    pub cache_levels: Vec<CacheLevelMetrics>,
    /// NUMA nodes utilization
    pub numa_nodes: Vec<NumaNodeMetrics>,
    /// GPU utilization (if available)
    pub gpu_utilization: Option<GpuMetrics>,
    /// Storage utilization
    pub storage_utilization: StorageMetrics,
}

/// Memory segment utilization
#[derive(Debug, Clone)]
pub struct MemorySegmentMetrics {
    /// Heap utilization
    pub heap_utilization: f64,
    /// Stack utilization
    pub stack_utilization: f64,
    /// Shared memory utilization
    pub shared_memory_utilization: f64,
    /// Memory fragmentation
    pub fragmentation: f64,
}

/// Cache level performance metrics
#[derive(Debug, Clone)]
pub struct CacheLevelMetrics {
    /// Cache level (L1, L2, L3)
    pub level: u8,
    /// Hit rate
    pub hit_rate: f64,
    /// Miss rate
    pub miss_rate: f64,
    /// Utilization
    pub utilization: f64,
    /// Bandwidth usage
    pub bandwidth_usage: f64,
}

/// NUMA node performance metrics
#[derive(Debug, Clone)]
pub struct NumaNodeMetrics {
    /// Node ID
    pub node_id: usize,
    /// Memory utilization
    pub memory_utilization: f64,
    /// CPU utilization
    pub cpu_utilization: f64,
    /// Inter-node traffic
    pub inter_node_traffic: u64,
    /// Local vs remote access ratio
    pub locality_ratio: f64,
}

/// GPU performance metrics
#[derive(Debug, Clone)]
pub struct GpuMetrics {
    /// GPU utilization
    pub gpu_utilization: f64,
    /// Memory utilization
    pub memory_utilization: f64,
    /// Temperature
    pub temperature: f64,
    /// Power consumption
    pub power_consumption: f64,
    /// Compute capability utilization
    pub compute_utilization: f64,
}

/// Storage performance metrics
#[derive(Debug, Clone)]
pub struct StorageMetrics {
    /// Read bandwidth (bytes/sec)
    pub read_bandwidth: u64,
    /// Write bandwidth (bytes/sec)
    pub write_bandwidth: u64,
    /// IOPS (operations/sec)
    pub iops: u64,
    /// Queue depth
    pub queue_depth: f64,
    /// Latency (milliseconds)
    pub latency_ms: f64,
}

/// Custom metric definition
#[derive(Debug, Clone)]
pub struct CustomMetric {
    /// Metric name
    pub name: String,
    /// Metric value
    pub value: f64,
    /// Metric unit
    pub unit: String,
    /// Collection timestamp
    pub timestamp: SystemTime,
    /// Metric metadata
    pub metadata: HashMap<String, String>,
}

/// Configuration for monitoring system
#[derive(Debug, Clone)]
pub struct MonitoringConfig {
    /// Metrics collection interval (seconds)
    pub collection_interval: Duration,
    /// Maximum metrics history size
    pub max_history_size: usize,
    /// Enable detailed resource monitoring
    pub enable_detailed_monitoring: bool,
    /// Enable predictive analytics
    pub enable_prediction: bool,
    /// Alert notification settings
    pub enable_alerts: bool,
    /// Dashboard update interval
    pub dashboard_update_interval: Duration,
    /// Custom metric definitions
    pub custom_metrics: Vec<CustomMetricDefinition>,
}

/// Custom metric definition for configuration
#[derive(Debug, Clone)]
pub struct CustomMetricDefinition {
    /// Metric name
    pub name: String,
    /// Metric unit
    pub unit: String,
    /// Collection function name
    pub collector: String,
    /// Collection interval
    pub interval: Duration,
}

impl MetricsCollector {
    /// Create new metrics collector
    pub(crate) fn new() -> Self {
        Self {
            metrics_history: VecDeque::with_capacity(10000),
            current_metrics: SystemMetrics::default(),
            operation_metrics: HashMap::new(),
            resource_metrics: ResourceMetrics::default(),
            custom_metrics: HashMap::new(),
        }
    }

    /// Add performance snapshot to history
    pub(crate) fn add_snapshot(&mut self, snapshot: PerformanceSnapshot) {
        if self.metrics_history.len() >= 10000 {
            self.metrics_history.pop_front();
        }
        self.metrics_history.push_back(snapshot);
    }

    /// Get recent metrics history
    pub(crate) fn get_recent_history(&self, count: usize) -> Vec<&PerformanceSnapshot> {
        self.metrics_history.iter().rev().take(count).collect()
    }

    /// Update operation metrics
    pub(crate) fn update_operation_metrics(
        &mut self,
        operation: String,
        metrics: OperationMetrics,
    ) {
        self.operation_metrics.insert(operation, metrics);
    }

    /// Add custom metric
    pub(crate) fn add_custom_metric(&mut self, metric: CustomMetric) {
        self.custom_metrics.insert(metric.name.clone(), metric);
    }
}

impl Default for SystemMetrics {
    fn default() -> Self {
        Self {
            total_ops_per_second: 0.0,
            average_latency_ms: 0.0,
            memory_usage: 0,
            cpu_utilization: 0.0,
            cache_hit_rate: 0.0,
            network_bandwidth: 0,
            disk_io_rate: 0,
            error_rate: 0.0,
        }
    }
}

impl Default for MemorySegmentMetrics {
    fn default() -> Self {
        Self {
            heap_utilization: 0.0,
            stack_utilization: 0.0,
            shared_memory_utilization: 0.0,
            fragmentation: 0.0,
        }
    }
}

impl Default for StorageMetrics {
    fn default() -> Self {
        Self {
            read_bandwidth: 0,
            write_bandwidth: 0,
            iops: 0,
            queue_depth: 0.0,
            latency_ms: 0.0,
        }
    }
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            collection_interval: Duration::from_secs(1),
            max_history_size: 10000,
            enable_detailed_monitoring: true,
            enable_prediction: true,
            enable_alerts: true,
            dashboard_update_interval: Duration::from_secs(5),
            custom_metrics: Vec::new(),
        }
    }
}
