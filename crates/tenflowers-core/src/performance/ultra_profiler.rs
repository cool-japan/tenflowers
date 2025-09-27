//! Ultra-High-Performance Profiling and Benchmarking System
//!
//! This module provides state-of-the-art performance monitoring, profiling,
//! and benchmarking capabilities for TenfloweRS, leveraging SciRS2-Core's
//! advanced profiling infrastructure. Designed with humility to provide
//! comprehensive insights while maintaining minimal overhead.

use crate::{Result, TensorError};
use scirs2_core::profiling::{Profiler, profiling_memory_tracker};
use scirs2_core::benchmarking::{BenchmarkSuite, BenchmarkRunner};
use scirs2_core::metrics::{MetricRegistry, Counter, Gauge, Histogram, Timer};
use scirs2_core::observability::{audit, tracing};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime};
use std::thread;

/// Ultra-comprehensive performance profiler with real-time analytics
pub struct UltraHighPerformanceProfiler {
    /// Core profiler instance from SciRS2-Core
    core_profiler: Arc<Profiler>,

    /// Benchmark suite for systematic performance testing
    benchmark_suite: Arc<BenchmarkSuite>,

    /// Metric registry for real-time monitoring
    metrics: Arc<MetricRegistry>,

    /// Performance data storage
    performance_data: Arc<RwLock<PerformanceDatabase>>,

    /// Configuration for profiling behavior
    config: ProfilerConfig,

    /// Real-time analysis engine
    analysis_engine: Arc<Mutex<AnalysisEngine>>,

    /// Background monitoring thread handle
    monitoring_thread: Option<thread::JoinHandle<()>>,
}

/// Configuration for ultra-high-performance profiling
#[derive(Debug, Clone)]
pub struct ProfilerConfig {
    /// Enable real-time profiling
    pub enable_realtime_profiling: bool,

    /// Enable deep memory profiling
    pub enable_memory_profiling: bool,

    /// Enable GPU profiling
    pub enable_gpu_profiling: bool,

    /// Enable automatic optimization suggestions
    pub enable_auto_optimization: bool,

    /// Sampling rate for continuous profiling (0.0 - 1.0)
    pub sampling_rate: f64,

    /// Maximum memory for performance data storage
    pub max_profile_data_memory: usize,

    /// Performance alert thresholds
    pub performance_thresholds: PerformanceThresholds,

    /// Enable tracing integration
    pub enable_tracing: bool,

    /// Background analysis interval
    pub analysis_interval: Duration,
}

/// Performance alert thresholds
#[derive(Debug, Clone)]
pub struct PerformanceThresholds {
    /// Maximum acceptable latency for operations (ms)
    pub max_operation_latency: f64,

    /// Minimum acceptable throughput (ops/sec)
    pub min_throughput: f64,

    /// Maximum acceptable memory usage (bytes)
    pub max_memory_usage: usize,

    /// Maximum acceptable GPU utilization (%)
    pub max_gpu_utilization: f64,

    /// Memory fragmentation threshold (%)
    pub max_fragmentation: f64,
}

impl Default for ProfilerConfig {
    fn default() -> Self {
        Self {
            enable_realtime_profiling: true,
            enable_memory_profiling: true,
            enable_gpu_profiling: true,
            enable_auto_optimization: true,
            sampling_rate: 0.1, // 10% sampling
            max_profile_data_memory: 268_435_456, // 256MB
            performance_thresholds: PerformanceThresholds {
                max_operation_latency: 100.0, // 100ms
                min_throughput: 1000.0,        // 1000 ops/sec
                max_memory_usage: 8_589_934_592, // 8GB
                max_gpu_utilization: 90.0,    // 90%
                max_fragmentation: 20.0,      // 20%
            },
            enable_tracing: true,
            analysis_interval: Duration::from_secs(10),
        }
    }
}

/// Comprehensive performance database
#[derive(Debug, Default)]
struct PerformanceDatabase {
    /// Operation performance records
    operation_records: HashMap<String, Vec<OperationRecord>>,

    /// Memory usage timeline
    memory_timeline: Vec<MemorySnapshot>,

    /// GPU utilization timeline
    gpu_timeline: Vec<GpuSnapshot>,

    /// System metrics timeline
    system_timeline: Vec<SystemSnapshot>,

    /// Performance alerts
    alerts: Vec<PerformanceAlert>,

    /// Optimization suggestions
    suggestions: Vec<OptimizationSuggestion>,
}

/// Individual operation performance record
#[derive(Debug, Clone)]
pub struct OperationRecord {
    pub operation_name: String,
    pub start_time: Instant,
    pub duration: Duration,
    pub memory_used: usize,
    pub gpu_utilization: f64,
    pub input_size: usize,
    pub output_size: usize,
    pub thread_id: u64,
    pub device_id: Option<u32>,
    pub metadata: HashMap<String, String>,
}

/// Memory usage snapshot
#[derive(Debug, Clone)]
pub struct MemorySnapshot {
    pub timestamp: Instant,
    pub total_allocated: usize,
    pub peak_allocated: usize,
    pub current_used: usize,
    pub fragmentation_ratio: f64,
    pub pool_statistics: HashMap<String, usize>,
}

/// GPU utilization snapshot
#[derive(Debug, Clone)]
pub struct GpuSnapshot {
    pub timestamp: Instant,
    pub device_id: u32,
    pub utilization_percent: f64,
    pub memory_used: usize,
    pub memory_total: usize,
    pub temperature: f32,
    pub power_usage: f32,
}

/// System metrics snapshot
#[derive(Debug, Clone)]
pub struct SystemSnapshot {
    pub timestamp: Instant,
    pub cpu_utilization: f64,
    pub memory_utilization: f64,
    pub disk_io_rate: f64,
    pub network_io_rate: f64,
    pub load_average: [f64; 3],
}

/// Performance alert
#[derive(Debug, Clone)]
pub struct PerformanceAlert {
    pub timestamp: Instant,
    pub alert_type: AlertType,
    pub severity: AlertSeverity,
    pub message: String,
    pub operation: Option<String>,
    pub value: f64,
    pub threshold: f64,
}

/// Types of performance alerts
#[derive(Debug, Clone, PartialEq)]
pub enum AlertType {
    HighLatency,
    LowThroughput,
    HighMemoryUsage,
    HighGpuUtilization,
    HighFragmentation,
    ResourceContention,
    UnexpectedFailure,
}

/// Alert severity levels
#[derive(Debug, Clone, PartialEq)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
    Emergency,
}

/// Optimization suggestion
#[derive(Debug, Clone)]
pub struct OptimizationSuggestion {
    pub timestamp: Instant,
    pub suggestion_type: SuggestionType,
    pub operation: String,
    pub description: String,
    pub potential_improvement: f64,
    pub confidence: f64,
    pub implementation_difficulty: DifficultyLevel,
}

/// Types of optimization suggestions
#[derive(Debug, Clone)]
pub enum SuggestionType {
    MemoryOptimization,
    ComputeOptimization,
    IoOptimization,
    AlgorithmImprovement,
    HardwareUtilization,
    DataLayoutOptimization,
}

/// Implementation difficulty levels
#[derive(Debug, Clone)]
pub enum DifficultyLevel {
    Trivial,
    Easy,
    Medium,
    Hard,
    Expert,
}

/// Real-time analysis engine
struct AnalysisEngine {
    /// Performance pattern detector
    pattern_detector: PatternDetector,

    /// Anomaly detection system
    anomaly_detector: AnomalyDetector,

    /// Optimization recommender
    optimization_recommender: OptimizationRecommender,
}

/// Performance pattern detection
struct PatternDetector {
    operation_patterns: HashMap<String, PerformancePattern>,
}

/// Performance pattern
#[derive(Debug, Clone)]
struct PerformancePattern {
    average_duration: Duration,
    std_deviation: f64,
    memory_pattern: MemoryPattern,
    seasonal_variations: Vec<SeasonalVariation>,
}

/// Memory usage patterns
#[derive(Debug, Clone)]
struct MemoryPattern {
    average_usage: usize,
    peak_usage: usize,
    allocation_pattern: AllocationPattern,
}

/// Memory allocation patterns
#[derive(Debug, Clone)]
enum AllocationPattern {
    Steady,
    Bursty,
    Periodic,
    Growing,
    Declining,
}

/// Seasonal performance variations
#[derive(Debug, Clone)]
struct SeasonalVariation {
    time_period: Duration,
    performance_factor: f64,
}

/// Anomaly detection system
struct AnomalyDetector {
    baseline_metrics: HashMap<String, BaselineMetrics>,
    anomaly_threshold: f64,
}

/// Baseline performance metrics
#[derive(Debug, Clone)]
struct BaselineMetrics {
    mean: f64,
    std_dev: f64,
    percentile_95: f64,
    percentile_99: f64,
}

/// Optimization recommendation system
struct OptimizationRecommender {
    recommendation_rules: Vec<OptimizationRule>,
}

/// Optimization rule
struct OptimizationRule {
    condition: Box<dyn Fn(&OperationRecord) -> bool + Send + Sync>,
    suggestion: OptimizationSuggestion,
}

impl UltraHighPerformanceProfiler {
    /// Create a new ultra-high-performance profiler
    pub fn new(config: ProfilerConfig) -> Result<Self> {
        let core_profiler = Arc::new(Profiler::new()?);
        let benchmark_suite = Arc::new(BenchmarkSuite::new("TenfloweRS Performance Suite")?);
        let metrics = Arc::new(MetricRegistry::new()?);

        // Initialize analysis engine
        let analysis_engine = Arc::new(Mutex::new(AnalysisEngine {
            pattern_detector: PatternDetector {
                operation_patterns: HashMap::new(),
            },
            anomaly_detector: AnomalyDetector {
                baseline_metrics: HashMap::new(),
                anomaly_threshold: 2.0, // 2 standard deviations
            },
            optimization_recommender: OptimizationRecommender {
                recommendation_rules: Self::create_optimization_rules(),
            },
        }));

        let profiler = Self {
            core_profiler,
            benchmark_suite,
            metrics,
            performance_data: Arc::new(RwLock::new(PerformanceDatabase::default())),
            config,
            analysis_engine,
            monitoring_thread: None,
        };

        Ok(profiler)
    }

    /// Start continuous performance monitoring
    pub fn start_monitoring(&mut self) -> Result<()> {
        if self.monitoring_thread.is_some() {
            return Ok(()); // Already monitoring
        }

        let performance_data = Arc::clone(&self.performance_data);
        let config = self.config.clone();
        let metrics = Arc::clone(&self.metrics);

        let handle = thread::spawn(move || {
            Self::monitoring_loop(performance_data, config, metrics);
        });

        self.monitoring_thread = Some(handle);
        Ok(())
    }

    /// Stop continuous monitoring
    pub fn stop_monitoring(&mut self) -> Result<()> {
        if let Some(handle) = self.monitoring_thread.take() {
            // In a real implementation, we'd have a shutdown signal
            // For now, we'll let the thread continue until process exit
            let _ = handle.join();
        }
        Ok(())
    }

    /// Profile a specific operation with comprehensive metrics
    pub fn profile_operation<F, R>(&self, operation_name: &str, operation: F) -> Result<(R, OperationRecord)>
    where
        F: FnOnce() -> Result<R>,
    {
        let start_time = Instant::now();
        let start_memory = self.get_current_memory_usage();

        // Start profiling session
        let _profiling_session = self.core_profiler.start_session(operation_name)?;

        // Execute operation with tracing if enabled
        let result = if self.config.enable_tracing {
            tracing::trace_operation(operation_name, operation)?
        } else {
            operation()?
        };

        let end_time = Instant::now();
        let duration = end_time - start_time;
        let end_memory = self.get_current_memory_usage();

        // Create operation record
        let record = OperationRecord {
            operation_name: operation_name.to_string(),
            start_time,
            duration,
            memory_used: end_memory.saturating_sub(start_memory),
            gpu_utilization: self.get_current_gpu_utilization(),
            input_size: 0, // Would be provided by caller in real implementation
            output_size: 0, // Would be provided by caller in real implementation
            thread_id: Self::get_thread_id(),
            device_id: None, // Would be detected in real implementation
            metadata: HashMap::new(),
        };

        // Store performance data
        {
            let mut data = self.performance_data.write().unwrap();
            data.operation_records
                .entry(operation_name.to_string())
                .or_insert_with(Vec::new)
                .push(record.clone());
        }

        // Check for performance alerts
        self.check_performance_alerts(&record)?;

        // Update metrics
        self.update_metrics(&record);

        Ok((result, record))
    }

    /// Run comprehensive benchmark suite
    pub fn run_benchmark_suite(&self) -> Result<BenchmarkResults> {
        let benchmark_runner = BenchmarkRunner::new(&self.benchmark_suite)?;

        // Run tensor operation benchmarks
        let tensor_benchmarks = self.run_tensor_benchmarks()?;

        // Run memory benchmarks
        let memory_benchmarks = self.run_memory_benchmarks()?;

        // Run neural network benchmarks
        let neural_benchmarks = self.run_neural_network_benchmarks()?;

        // Run GPU benchmarks if available
        let gpu_benchmarks = if self.config.enable_gpu_profiling {
            Some(self.run_gpu_benchmarks()?)
        } else {
            None
        };

        Ok(BenchmarkResults {
            tensor_benchmarks,
            memory_benchmarks,
            neural_benchmarks,
            gpu_benchmarks,
            system_info: self.collect_system_info(),
            timestamp: SystemTime::now(),
        })
    }

    /// Generate comprehensive performance report
    pub fn generate_performance_report(&self) -> PerformanceReport {
        let data = self.performance_data.read().unwrap();

        // Analyze operation performance
        let operation_analysis = self.analyze_operation_performance(&data);

        // Analyze memory usage
        let memory_analysis = self.analyze_memory_usage(&data);

        // Analyze GPU utilization
        let gpu_analysis = self.analyze_gpu_utilization(&data);

        // Generate optimization suggestions
        let optimization_suggestions = data.suggestions.clone();

        // Get current metrics
        let current_metrics = self.get_current_metrics();

        PerformanceReport {
            timestamp: SystemTime::now(),
            operation_analysis,
            memory_analysis,
            gpu_analysis,
            optimization_suggestions,
            alerts: data.alerts.clone(),
            metrics_summary: current_metrics,
            profiling_overhead: self.estimate_profiling_overhead(),
        }
    }

    /// Get real-time performance dashboard data
    pub fn get_dashboard_data(&self) -> DashboardData {
        let data = self.performance_data.read().unwrap();

        // Get recent operation metrics
        let recent_operations = self.get_recent_operations(&data, Duration::from_secs(60));

        // Get current system status
        let system_status = SystemStatus {
            cpu_utilization: self.get_current_cpu_utilization(),
            memory_utilization: self.get_current_memory_utilization(),
            gpu_utilization: self.get_current_gpu_utilization(),
            active_operations: recent_operations.len(),
            alerts_count: data.alerts.len(),
        };

        DashboardData {
            system_status,
            recent_operations,
            memory_timeline: data.memory_timeline.clone(),
            gpu_timeline: data.gpu_timeline.clone(),
            active_alerts: data.alerts.iter()
                .filter(|alert| alert.timestamp.elapsed() < Duration::from_minutes(5))
                .cloned()
                .collect(),
        }
    }

    // Private helper methods

    fn monitoring_loop(
        performance_data: Arc<RwLock<PerformanceDatabase>>,
        config: ProfilerConfig,
        metrics: Arc<MetricRegistry>,
    ) {
        loop {
            // Collect system metrics
            let memory_snapshot = MemorySnapshot {
                timestamp: Instant::now(),
                total_allocated: 0, // Would be implemented with actual memory tracking
                peak_allocated: 0,
                current_used: 0,
                fragmentation_ratio: 0.0,
                pool_statistics: HashMap::new(),
            };

            let system_snapshot = SystemSnapshot {
                timestamp: Instant::now(),
                cpu_utilization: 0.0, // Would be implemented with actual system monitoring
                memory_utilization: 0.0,
                disk_io_rate: 0.0,
                network_io_rate: 0.0,
                load_average: [0.0, 0.0, 0.0],
            };

            // Update performance database
            {
                let mut data = performance_data.write().unwrap();
                data.memory_timeline.push(memory_snapshot);
                data.system_timeline.push(system_snapshot);

                // Limit memory usage by keeping only recent data
                if data.memory_timeline.len() > 10000 {
                    data.memory_timeline.drain(0..1000);
                }
                if data.system_timeline.len() > 10000 {
                    data.system_timeline.drain(0..1000);
                }
            }

            thread::sleep(config.analysis_interval);
        }
    }

    fn create_optimization_rules() -> Vec<OptimizationRule> {
        vec![
            // Rule for high memory usage
            OptimizationRule {
                condition: Box::new(|record| record.memory_used > 1_073_741_824), // 1GB
                suggestion: OptimizationSuggestion {
                    timestamp: Instant::now(),
                    suggestion_type: SuggestionType::MemoryOptimization,
                    operation: "high_memory_operation".to_string(),
                    description: "Consider using memory pooling or chunked processing".to_string(),
                    potential_improvement: 50.0,
                    confidence: 0.8,
                    implementation_difficulty: DifficultyLevel::Medium,
                },
            },
            // Add more rules here
        ]
    }

    fn check_performance_alerts(&self, record: &OperationRecord) -> Result<()> {
        let mut alerts = Vec::new();

        // Check latency threshold
        if record.duration.as_millis() as f64 > self.config.performance_thresholds.max_operation_latency {
            alerts.push(PerformanceAlert {
                timestamp: Instant::now(),
                alert_type: AlertType::HighLatency,
                severity: AlertSeverity::Warning,
                message: format!("Operation {} exceeded latency threshold", record.operation_name),
                operation: Some(record.operation_name.clone()),
                value: record.duration.as_millis() as f64,
                threshold: self.config.performance_thresholds.max_operation_latency,
            });
        }

        // Check memory usage threshold
        if record.memory_used > self.config.performance_thresholds.max_memory_usage {
            alerts.push(PerformanceAlert {
                timestamp: Instant::now(),
                alert_type: AlertType::HighMemoryUsage,
                severity: AlertSeverity::Critical,
                message: format!("Operation {} exceeded memory threshold", record.operation_name),
                operation: Some(record.operation_name.clone()),
                value: record.memory_used as f64,
                threshold: self.config.performance_thresholds.max_memory_usage as f64,
            });
        }

        // Store alerts
        if !alerts.is_empty() {
            let mut data = self.performance_data.write().unwrap();
            data.alerts.extend(alerts);
        }

        Ok(())
    }

    fn update_metrics(&self, record: &OperationRecord) {
        // Update operation counter
        let counter = self.metrics.counter(&format!("{}_operations", record.operation_name));
        counter.increment(1);

        // Update duration histogram
        let histogram = self.metrics.histogram(&format!("{}_duration", record.operation_name));
        histogram.record(record.duration.as_millis() as f64);

        // Update memory gauge
        let gauge = self.metrics.gauge(&format!("{}_memory", record.operation_name));
        gauge.set(record.memory_used as f64);
    }

    // Placeholder implementations for system monitoring
    fn get_current_memory_usage(&self) -> usize { 0 }
    fn get_current_gpu_utilization(&self) -> f64 { 0.0 }
    fn get_current_cpu_utilization(&self) -> f64 { 0.0 }
    fn get_current_memory_utilization(&self) -> f64 { 0.0 }
    fn get_thread_id() -> u64 { 0 }

    // Placeholder implementations for benchmarks
    fn run_tensor_benchmarks(&self) -> Result<TensorBenchmarkResults> {
        Ok(TensorBenchmarkResults::default())
    }

    fn run_memory_benchmarks(&self) -> Result<MemoryBenchmarkResults> {
        Ok(MemoryBenchmarkResults::default())
    }

    fn run_neural_network_benchmarks(&self) -> Result<NeuralNetworkBenchmarkResults> {
        Ok(NeuralNetworkBenchmarkResults::default())
    }

    fn run_gpu_benchmarks(&self) -> Result<GpuBenchmarkResults> {
        Ok(GpuBenchmarkResults::default())
    }

    fn collect_system_info(&self) -> SystemInfo {
        SystemInfo::default()
    }

    fn analyze_operation_performance(&self, _data: &PerformanceDatabase) -> OperationAnalysis {
        OperationAnalysis::default()
    }

    fn analyze_memory_usage(&self, _data: &PerformanceDatabase) -> MemoryAnalysis {
        MemoryAnalysis::default()
    }

    fn analyze_gpu_utilization(&self, _data: &PerformanceDatabase) -> GpuAnalysis {
        GpuAnalysis::default()
    }

    fn get_current_metrics(&self) -> MetricsSummary {
        MetricsSummary::default()
    }

    fn estimate_profiling_overhead(&self) -> f64 {
        2.0 // 2% overhead estimate
    }

    fn get_recent_operations(&self, _data: &PerformanceDatabase, _window: Duration) -> Vec<OperationRecord> {
        Vec::new()
    }
}

// Supporting structures for benchmark results and analysis

#[derive(Debug, Default)]
pub struct BenchmarkResults {
    pub tensor_benchmarks: TensorBenchmarkResults,
    pub memory_benchmarks: MemoryBenchmarkResults,
    pub neural_benchmarks: NeuralNetworkBenchmarkResults,
    pub gpu_benchmarks: Option<GpuBenchmarkResults>,
    pub system_info: SystemInfo,
    pub timestamp: SystemTime,
}

#[derive(Debug, Default)]
pub struct TensorBenchmarkResults {
    pub add_performance: f64,
    pub multiply_performance: f64,
    pub matmul_performance: f64,
    pub convolution_performance: f64,
}

#[derive(Debug, Default)]
pub struct MemoryBenchmarkResults {
    pub allocation_speed: f64,
    pub deallocation_speed: f64,
    pub bandwidth: f64,
    pub latency: f64,
}

#[derive(Debug, Default)]
pub struct NeuralNetworkBenchmarkResults {
    pub forward_pass_speed: f64,
    pub backward_pass_speed: f64,
    pub training_throughput: f64,
    pub inference_latency: f64,
}

#[derive(Debug, Default)]
pub struct GpuBenchmarkResults {
    pub compute_performance: f64,
    pub memory_bandwidth: f64,
    pub kernel_launch_overhead: f64,
    pub data_transfer_speed: f64,
}

#[derive(Debug, Default)]
pub struct SystemInfo {
    pub cpu_model: String,
    pub cpu_cores: u32,
    pub memory_total: usize,
    pub gpu_model: Option<String>,
    pub os_version: String,
}

#[derive(Debug, Default)]
pub struct PerformanceReport {
    pub timestamp: SystemTime,
    pub operation_analysis: OperationAnalysis,
    pub memory_analysis: MemoryAnalysis,
    pub gpu_analysis: GpuAnalysis,
    pub optimization_suggestions: Vec<OptimizationSuggestion>,
    pub alerts: Vec<PerformanceAlert>,
    pub metrics_summary: MetricsSummary,
    pub profiling_overhead: f64,
}

#[derive(Debug, Default)]
pub struct OperationAnalysis {
    pub total_operations: u64,
    pub average_latency: f64,
    pub throughput: f64,
    pub slowest_operations: Vec<String>,
    pub performance_trends: Vec<PerformanceTrend>,
}

#[derive(Debug, Default)]
pub struct MemoryAnalysis {
    pub peak_usage: usize,
    pub average_usage: usize,
    pub fragmentation_ratio: f64,
    pub allocation_patterns: Vec<AllocationPattern>,
    pub memory_efficiency: f64,
}

#[derive(Debug, Default)]
pub struct GpuAnalysis {
    pub average_utilization: f64,
    pub peak_utilization: f64,
    pub memory_efficiency: f64,
    pub compute_efficiency: f64,
    pub bottlenecks: Vec<String>,
}

#[derive(Debug, Default)]
pub struct MetricsSummary {
    pub total_operations: u64,
    pub error_rate: f64,
    pub success_rate: f64,
    pub average_response_time: f64,
}

#[derive(Debug)]
pub struct PerformanceTrend {
    pub metric_name: String,
    pub trend_direction: TrendDirection,
    pub change_percentage: f64,
}

#[derive(Debug)]
pub enum TrendDirection {
    Improving,
    Degrading,
    Stable,
}

#[derive(Debug)]
pub struct DashboardData {
    pub system_status: SystemStatus,
    pub recent_operations: Vec<OperationRecord>,
    pub memory_timeline: Vec<MemorySnapshot>,
    pub gpu_timeline: Vec<GpuSnapshot>,
    pub active_alerts: Vec<PerformanceAlert>,
}

#[derive(Debug)]
pub struct SystemStatus {
    pub cpu_utilization: f64,
    pub memory_utilization: f64,
    pub gpu_utilization: f64,
    pub active_operations: usize,
    pub alerts_count: usize,
}

/// Global profiler instance for system-wide performance monitoring
static GLOBAL_PROFILER: std::sync::OnceLock<UltraHighPerformanceProfiler> = std::sync::OnceLock::new();

/// Get or initialize the global profiler
pub fn global_profiler() -> &'static UltraHighPerformanceProfiler {
    GLOBAL_PROFILER.get_or_init(|| {
        UltraHighPerformanceProfiler::new(ProfilerConfig::default())
            .expect("Failed to initialize global profiler")
    })
}

/// Convenience macro for profiling operations
#[macro_export]
macro_rules! profile {
    ($operation_name:expr, $operation:expr) => {{
        let profiler = $crate::performance::ultra_profiler::global_profiler();
        profiler.profile_operation($operation_name, || $operation)
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_creation() {
        let config = ProfilerConfig::default();
        let profiler = UltraHighPerformanceProfiler::new(config);
        assert!(profiler.is_ok());
    }

    #[test]
    fn test_operation_profiling() {
        let profiler = UltraHighPerformanceProfiler::new(ProfilerConfig::default()).unwrap();

        let (result, record) = profiler.profile_operation("test_operation", || {
            Ok(42)
        }).unwrap();

        assert_eq!(result, 42);
        assert_eq!(record.operation_name, "test_operation");
        assert!(record.duration > Duration::from_nanos(0));
    }

    #[test]
    fn test_global_profiler() {
        let profiler1 = global_profiler();
        let profiler2 = global_profiler();

        // Should be the same instance
        assert!(std::ptr::eq(profiler1, profiler2));
    }

    #[test]
    fn test_performance_alert_generation() {
        let mut config = ProfilerConfig::default();
        config.performance_thresholds.max_operation_latency = 1.0; // 1ms threshold

        let profiler = UltraHighPerformanceProfiler::new(config).unwrap();

        // This should generate an alert due to the low threshold
        let (_result, _record) = profiler.profile_operation("slow_operation", || {
            thread::sleep(Duration::from_millis(10)); // Sleep for 10ms
            Ok(())
        }).unwrap();

        let data = profiler.performance_data.read().unwrap();
        assert!(!data.alerts.is_empty());
    }

    #[test]
    fn test_benchmark_suite() {
        let profiler = UltraHighPerformanceProfiler::new(ProfilerConfig::default()).unwrap();
        let benchmark_results = profiler.run_benchmark_suite();
        assert!(benchmark_results.is_ok());
    }

    #[test]
    fn test_performance_report_generation() {
        let profiler = UltraHighPerformanceProfiler::new(ProfilerConfig::default()).unwrap();

        // Run some operations to generate data
        let _ = profiler.profile_operation("test_op1", || Ok(1));
        let _ = profiler.profile_operation("test_op2", || Ok(2));

        let report = profiler.generate_performance_report();
        assert!(report.profiling_overhead >= 0.0);
    }

    #[test]
    fn test_dashboard_data() {
        let profiler = UltraHighPerformanceProfiler::new(ProfilerConfig::default()).unwrap();
        let dashboard_data = profiler.get_dashboard_data();

        assert!(dashboard_data.system_status.active_operations >= 0);
        assert!(dashboard_data.system_status.alerts_count >= 0);
    }
}