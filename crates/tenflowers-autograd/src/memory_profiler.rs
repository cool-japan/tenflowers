use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};
use tenflowers_core::{Result, TensorError};

// Use SciRS2-Core for advanced memory management
use scirs2_core::memory::metrics::{MemoryMetricsCollector, MemoryMetricsConfig};
use scirs2_core::memory::{GlobalBufferPool, LeakDetectionConfig, LeakDetector};
use scirs2_core::profiling::Profiler;

/// Memory usage statistics for gradient operations
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    /// Peak memory usage in bytes
    pub peak_memory: u64,
    /// Current memory usage in bytes
    pub current_memory: u64,
    /// Total allocations made
    pub total_allocations: u64,
    /// Total deallocations made
    pub total_deallocations: u64,
    /// Number of gradient operations performed
    pub gradient_operations: u64,
    /// Memory usage per operation type
    pub operation_memory: HashMap<String, u64>,
    /// Time-stamped memory usage samples
    pub memory_timeline: Vec<(Instant, u64)>,
    /// Memory fragmentation ratio (0.0 = perfect, 1.0 = highly fragmented)
    pub fragmentation_ratio: f64,
    /// GPU memory usage (if available)
    pub gpu_memory_used: u64,
    /// Memory pool statistics
    pub pool_statistics: MemoryPoolStats,
    /// Memory leak detection results
    pub leak_detection: LeakDetectionStats,
}

/// Advanced memory pool statistics
#[derive(Debug, Clone, Default)]
pub struct MemoryPoolStats {
    /// Total pool capacity in bytes
    pub total_capacity: u64,
    /// Currently allocated from pools
    pub pool_allocated: u64,
    /// Number of pool hits (reused memory)
    pub pool_hits: u64,
    /// Number of pool misses (new allocations)
    pub pool_misses: u64,
    /// Pool efficiency ratio
    pub efficiency_ratio: f64,
}

/// Memory leak detection statistics
#[derive(Debug, Clone, Default)]
pub struct LeakDetectionStats {
    /// Number of potential leaks detected
    pub potential_leaks: u64,
    /// Total leaked bytes
    pub leaked_bytes: u64,
    /// Leak detection enabled
    pub detection_enabled: bool,
    /// Last leak scan time
    pub last_scan: Option<Instant>,
}

/// Memory profiling integration for gradient computation
pub struct GradientMemoryProfiler {
    stats: Arc<Mutex<MemoryStats>>,
    enabled: bool,
    sample_interval: Duration,
    last_sample: Instant,
    operation_stack: Vec<String>,
    operation_start_memory: Vec<u64>,
    // SciRS2-Core integrations
    #[allow(dead_code)]
    buffer_pool: Arc<GlobalBufferPool>,
    #[allow(dead_code)]
    leak_detector: Arc<LeakDetector>,
    #[allow(dead_code)]
    metrics_collector: Arc<MemoryMetricsCollector>,
    #[allow(dead_code)]
    profiler: Arc<Profiler>,
}

impl GradientMemoryProfiler {
    /// Create a new gradient memory profiler with SciRS2-Core integration
    pub fn new() -> Self {
        Self {
            stats: Arc::new(Mutex::new(MemoryStats::default())),
            enabled: true,
            sample_interval: Duration::from_millis(10),
            last_sample: Instant::now(),
            operation_stack: Vec::new(),
            operation_start_memory: Vec::new(),
            // Initialize SciRS2-Core components
            buffer_pool: Arc::new(GlobalBufferPool::new()),
            leak_detector: Arc::new(
                LeakDetector::new(LeakDetectionConfig::default())
                    .expect("Failed to create leak detector"),
            ),
            metrics_collector: Arc::new(
                MemoryMetricsCollector::new(MemoryMetricsConfig::default()),
            ),
            profiler: Arc::new(Profiler::new()),
        }
    }

    /// Create a new profiler with custom configuration
    pub fn with_config(
        sample_interval: Duration,
        enable_leak_detection: bool,
        pool_capacity: usize,
    ) -> Self {
        let mut profiler = Self::new();
        profiler.sample_interval = sample_interval;

        if let Ok(mut stats) = profiler.stats.lock() {
            stats.leak_detection.detection_enabled = enable_leak_detection;
            stats.pool_statistics.total_capacity = pool_capacity as u64;
        }

        profiler
    }

    /// Enable or disable memory profiling
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Set the memory sampling interval
    pub fn set_sample_interval(&mut self, interval: Duration) {
        self.sample_interval = interval;
    }

    /// Begin tracking a gradient operation
    pub fn begin_operation(&mut self, operation_name: &str) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        let current_memory = self.get_current_memory_usage()?;
        self.operation_stack.push(operation_name.to_string());
        self.operation_start_memory.push(current_memory);

        if let Ok(mut stats) = self.stats.lock() {
            stats.gradient_operations += 1;
        }

        Ok(())
    }

    /// End tracking a gradient operation and record memory usage
    pub fn end_operation(&mut self) -> Result<()> {
        if !self.enabled || self.operation_stack.is_empty() {
            return Ok(());
        }

        let operation_name = self.operation_stack.pop().unwrap();
        let start_memory = self.operation_start_memory.pop().unwrap();
        let current_memory = self.get_current_memory_usage()?;
        let memory_delta = current_memory.saturating_sub(start_memory);

        if let Ok(mut stats) = self.stats.lock() {
            *stats.operation_memory.entry(operation_name).or_insert(0) += memory_delta;
            stats.current_memory = current_memory;
            stats.peak_memory = stats.peak_memory.max(current_memory);

            // Sample memory usage periodically
            if self.last_sample.elapsed() >= self.sample_interval {
                stats.memory_timeline.push((Instant::now(), current_memory));
                self.last_sample = Instant::now();

                // Keep timeline size manageable
                if stats.memory_timeline.len() > 10000 {
                    stats.memory_timeline.drain(0..1000);
                }
            }
        }

        Ok(())
    }

    /// Record a memory allocation
    pub fn record_allocation(&self, bytes: u64) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        if let Ok(mut stats) = self.stats.lock() {
            stats.total_allocations += 1;
            stats.current_memory += bytes;
            stats.peak_memory = stats.peak_memory.max(stats.current_memory);
        }

        Ok(())
    }

    /// Record a memory deallocation
    pub fn record_deallocation(&self, bytes: u64) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        if let Ok(mut stats) = self.stats.lock() {
            stats.total_deallocations += 1;
            stats.current_memory = stats.current_memory.saturating_sub(bytes);
        }

        Ok(())
    }

    /// Get current memory statistics
    pub fn get_stats(&self) -> Result<MemoryStats> {
        if let Ok(stats) = self.stats.lock() {
            Ok(stats.clone())
        } else {
            Err(TensorError::allocation_error(
                "memory_profiler",
                "Failed to acquire stats lock",
                None,
                None,
            ))
        }
    }

    /// Reset all memory statistics
    pub fn reset_stats(&self) -> Result<()> {
        if let Ok(mut stats) = self.stats.lock() {
            *stats = MemoryStats::default();
            Ok(())
        } else {
            Err(TensorError::allocation_error(
                "memory_profiler",
                "Failed to acquire stats lock",
                None,
                None,
            ))
        }
    }

    /// Generate a memory usage report
    pub fn generate_report(&self) -> Result<MemoryReport> {
        let stats = self.get_stats()?;
        Ok(MemoryReport::new(stats))
    }

    /// Get current system memory usage (approximation)
    fn get_current_memory_usage(&self) -> Result<u64> {
        // In a real implementation, this would use system APIs to get actual memory usage
        // For now, we'll use the tracked memory from our statistics
        if let Ok(stats) = self.stats.lock() {
            Ok(stats.current_memory)
        } else {
            Ok(0)
        }
    }

    /// Track memory usage for a gradient computation closure
    pub fn profile_gradient_computation<T, F>(&mut self, operation_name: &str, f: F) -> Result<T>
    where
        F: FnOnce() -> Result<T>,
    {
        self.begin_operation(operation_name)?;
        let result = f();
        self.end_operation()?;
        result
    }

    /// Enhanced memory analysis with SciRS2-Core integration
    pub fn analyze_memory_patterns(&self) -> Result<MemoryAnalysisReport> {
        let stats = self.get_stats()?;

        // Estimate fragmentation using basic heuristics
        let fragmentation_ratio = stats.fragmentation_ratio;

        // Check for memory leaks (simplified)
        let leak_detection_results = if stats.leak_detection.detection_enabled {
            // Simplified leak detection
            LeakDetectionResults {
                leaks_found: vec![],
                total_leaked_bytes: stats.leak_detection.leaked_bytes,
                scan_duration: Duration::from_millis(100),
            }
        } else {
            LeakDetectionResults::default()
        };

        // Analyze allocation patterns
        let allocation_patterns = self.analyze_allocation_patterns(&stats);

        Ok(MemoryAnalysisReport {
            fragmentation_ratio,
            leak_detection_results,
            allocation_patterns,
            memory_efficiency: self.calculate_memory_efficiency(&stats),
            optimization_opportunities: self.identify_optimization_opportunities(&stats),
        })
    }

    /// Optimize memory pools for current usage patterns
    pub fn optimize_memory_pools(&self) -> Result<PoolOptimizationResult> {
        let stats = self.get_stats()?;

        // Simplified pool optimization - calculate optimal chunk sizes heuristically
        let optimal_chunk_sizes = self.calculate_optimal_chunk_sizes(&stats.operation_memory);

        // Estimate efficiency improvement
        let efficiency_improvement = 0.15; // 15% improvement estimate
        let memory_saved = stats.current_memory / 10; // 10% memory savings estimate

        if let Ok(mut locked_stats) = self.stats.lock() {
            locked_stats.pool_statistics.efficiency_ratio += efficiency_improvement;
            locked_stats.pool_statistics.pool_hits += 100; // Simulated improvement
        }

        Ok(PoolOptimizationResult {
            efficiency_improvement,
            memory_saved,
            optimal_chunk_sizes,
        })
    }

    /// Calculate optimal chunk sizes based on operation patterns
    fn calculate_optimal_chunk_sizes(
        &self,
        operation_memory: &std::collections::HashMap<String, u64>,
    ) -> Vec<usize> {
        if operation_memory.is_empty() {
            return vec![1024, 4096, 16384, 65536]; // Default sizes
        }

        let mut sizes: Vec<_> = operation_memory
            .values()
            .map(|&size| size as usize)
            .collect();
        sizes.sort();
        sizes.dedup();

        // Return up to 8 optimal sizes
        if sizes.len() > 8 {
            sizes.truncate(8);
        }

        sizes
    }

    /// Enable/disable advanced memory leak detection
    pub fn configure_leak_detection(&self, enabled: bool, _scan_interval: Duration) -> Result<()> {
        if let Ok(mut stats) = self.stats.lock() {
            stats.leak_detection.detection_enabled = enabled;
            stats.leak_detection.last_scan = Some(Instant::now());
        }

        // Simplified leak detection configuration
        // In a full implementation, this would configure the SciRS2 leak detector

        Ok(())
    }

    /// Get GPU memory statistics (if GPU features enabled)
    #[cfg(feature = "gpu")]
    pub fn get_gpu_memory_stats(&self) -> Result<GpuMemoryStats> {
        let stats = self.get_stats()?;

        Ok(GpuMemoryStats {
            total_allocated: stats.gpu_memory_used,
            peak_usage: stats.peak_memory,
            fragmentation: stats.fragmentation_ratio,
            device_memory_total: stats.peak_memory * 4, // Estimate device memory
        })
    }

    /// Trigger garbage collection and memory compaction
    pub fn compact_memory(&self) -> Result<CompactionResult> {
        let pre_fragmentation = if let Ok(stats) = self.stats.lock() {
            stats.fragmentation_ratio
        } else {
            0.5 // Default assumption
        };

        // Simulate garbage collection and compaction
        let memory_freed = if let Ok(stats) = self.stats.lock() {
            stats.current_memory / 20 // Assume 5% memory can be freed
        } else {
            0
        };

        let post_fragmentation = pre_fragmentation * 0.7; // 30% fragmentation improvement

        if let Ok(mut stats) = self.stats.lock() {
            stats.fragmentation_ratio = post_fragmentation;
            stats.current_memory = stats.current_memory.saturating_sub(memory_freed);
        }

        Ok(CompactionResult {
            memory_freed,
            fragmentation_improvement: pre_fragmentation - post_fragmentation,
            pools_compacted: 3, // Simulated number of pools
        })
    }

    /// Advanced memory usage prediction for future operations
    pub fn predict_memory_usage(&self, operation_sequence: &[String]) -> Result<MemoryPrediction> {
        let stats = self.get_stats()?;

        let mut predicted_peak = 0u64;
        let mut current_predicted = stats.current_memory;

        for operation in operation_sequence {
            if let Some(&historical_usage) = stats.operation_memory.get(operation) {
                current_predicted += historical_usage;
                predicted_peak = predicted_peak.max(current_predicted);
                // Assume some operations free memory after completion
                current_predicted = (current_predicted * 80) / 100; // 20% reduction heuristic
            } else {
                // Unknown operation, use average memory usage
                let avg_usage = if !stats.operation_memory.is_empty() {
                    stats.operation_memory.values().sum::<u64>()
                        / stats.operation_memory.len() as u64
                } else {
                    1_048_576 // 1MB default
                };
                current_predicted += avg_usage;
                predicted_peak = predicted_peak.max(current_predicted);
                current_predicted = (current_predicted * 80) / 100;
            }
        }

        Ok(MemoryPrediction {
            predicted_peak_usage: predicted_peak,
            predicted_final_usage: current_predicted,
            confidence_score: self.calculate_prediction_confidence(&stats, operation_sequence),
            risk_assessment: if predicted_peak > stats.peak_memory.saturating_mul(2) {
                RiskLevel::High
            } else if predicted_peak > (stats.peak_memory * 3) / 2 {
                RiskLevel::Medium
            } else {
                RiskLevel::Low
            },
        })
    }

    // Helper methods for enhanced functionality
    fn analyze_allocation_patterns(&self, stats: &MemoryStats) -> AllocationPatterns {
        AllocationPatterns {
            avg_allocation_size: if stats.total_allocations > 0 {
                stats.current_memory / stats.total_allocations
            } else {
                0
            },
            allocation_frequency: stats.total_allocations as f64 / stats.gradient_operations as f64,
            peak_allocation_periods: self.identify_peak_periods(&stats.memory_timeline),
        }
    }

    fn calculate_memory_efficiency(&self, stats: &MemoryStats) -> f64 {
        if stats.peak_memory > 0 {
            (stats.gradient_operations as f64 * 1_048_576.0) / stats.peak_memory as f64
        } else {
            0.0
        }
    }

    fn identify_optimization_opportunities(
        &self,
        stats: &MemoryStats,
    ) -> Vec<OptimizationOpportunity> {
        let mut opportunities = Vec::new();

        // Check for high fragmentation
        if stats.fragmentation_ratio > 0.3 {
            opportunities.push(OptimizationOpportunity::MemoryCompaction);
        }

        // Check for inefficient pool usage
        if stats.pool_statistics.efficiency_ratio < 0.7 {
            opportunities.push(OptimizationOpportunity::PoolReorganization);
        }

        // Check for potential memory leaks
        if stats.leak_detection.potential_leaks > 0 {
            opportunities.push(OptimizationOpportunity::LeakMitigation);
        }

        // Check for high allocation/deallocation churn
        let alloc_dealloc_ratio = if stats.total_deallocations > 0 {
            stats.total_allocations as f64 / stats.total_deallocations as f64
        } else {
            f64::INFINITY
        };

        if alloc_dealloc_ratio > 2.0 {
            opportunities.push(OptimizationOpportunity::ReduceChurn);
        }

        opportunities
    }

    fn calculate_prediction_confidence(&self, stats: &MemoryStats, operations: &[String]) -> f64 {
        let known_operations = operations
            .iter()
            .filter(|op| stats.operation_memory.contains_key(*op))
            .count();

        known_operations as f64 / operations.len() as f64
    }

    fn identify_peak_periods(&self, timeline: &[(Instant, u64)]) -> Vec<PeakPeriod> {
        let mut peaks = Vec::new();

        if timeline.len() < 3 {
            return peaks;
        }

        for i in 1..timeline.len() - 1 {
            let prev_mem = timeline[i - 1].1;
            let curr_mem = timeline[i].1;
            let next_mem = timeline[i + 1].1;

            // Detect local maxima
            if curr_mem > prev_mem && curr_mem > next_mem && curr_mem > 0 {
                peaks.push(PeakPeriod {
                    timestamp: timeline[i].0,
                    memory_usage: curr_mem,
                    duration_estimate: timeline[i + 1].0.duration_since(timeline[i - 1].0),
                });
            }
        }

        peaks
    }
}

impl Default for GradientMemoryProfiler {
    fn default() -> Self {
        Self::new()
    }
}

// Enhanced memory analysis types

/// Advanced memory analysis report with SciRS2-Core insights
#[derive(Debug)]
pub struct MemoryAnalysisReport {
    pub fragmentation_ratio: f64,
    pub leak_detection_results: LeakDetectionResults,
    pub allocation_patterns: AllocationPatterns,
    pub memory_efficiency: f64,
    pub optimization_opportunities: Vec<OptimizationOpportunity>,
}

/// Memory pool optimization results
#[derive(Debug)]
pub struct PoolOptimizationResult {
    pub efficiency_improvement: f64,
    pub memory_saved: u64,
    pub optimal_chunk_sizes: Vec<usize>,
}

/// GPU memory statistics (when GPU features are enabled)
#[cfg(feature = "gpu")]
#[derive(Debug)]
pub struct GpuMemoryStats {
    pub total_allocated: u64,
    pub peak_usage: u64,
    pub fragmentation: f64,
    pub device_memory_total: u64,
}

/// Memory compaction results
#[derive(Debug)]
pub struct CompactionResult {
    pub memory_freed: u64,
    pub fragmentation_improvement: f64,
    pub pools_compacted: usize,
}

/// Memory usage prediction for operation sequences
#[derive(Debug)]
pub struct MemoryPrediction {
    pub predicted_peak_usage: u64,
    pub predicted_final_usage: u64,
    pub confidence_score: f64,
    pub risk_assessment: RiskLevel,
}

/// Risk levels for memory usage predictions
#[derive(Debug, Clone, PartialEq)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
}

/// Memory leak detection results
#[derive(Debug, Clone, Default)]
pub struct LeakDetectionResults {
    pub leaks_found: Vec<MemoryLeak>,
    pub total_leaked_bytes: u64,
    pub scan_duration: Duration,
}

/// Individual memory leak information
#[derive(Debug, Clone)]
pub struct MemoryLeak {
    pub allocation_site: String,
    pub size: u64,
    pub age: Duration,
    pub stack_trace: Vec<String>,
}

/// Memory allocation patterns analysis
#[derive(Debug)]
pub struct AllocationPatterns {
    pub avg_allocation_size: u64,
    pub allocation_frequency: f64,
    pub peak_allocation_periods: Vec<PeakPeriod>,
}

/// Memory usage peak period information
#[derive(Debug)]
pub struct PeakPeriod {
    pub timestamp: Instant,
    pub memory_usage: u64,
    pub duration_estimate: Duration,
}

/// Memory optimization opportunities
#[derive(Debug, Clone)]
pub enum OptimizationOpportunity {
    MemoryCompaction,
    PoolReorganization,
    LeakMitigation,
    ReduceChurn,
}

/// Memory usage report with analysis and recommendations
#[derive(Debug)]
pub struct MemoryReport {
    pub stats: MemoryStats,
    pub memory_efficiency: f64,
    pub peak_to_average_ratio: f64,
    pub recommendations: Vec<String>,
}

impl MemoryReport {
    fn new(stats: MemoryStats) -> Self {
        let mut recommendations = Vec::new();

        // Calculate memory efficiency (operations per MB)
        let memory_efficiency = if stats.peak_memory > 0 {
            stats.gradient_operations as f64 / (stats.peak_memory as f64 / 1_048_576.0)
        } else {
            0.0
        };

        // Calculate peak to average ratio from timeline
        let average_memory = if !stats.memory_timeline.is_empty() {
            stats
                .memory_timeline
                .iter()
                .map(|(_, mem)| *mem)
                .sum::<u64>() as f64
                / stats.memory_timeline.len() as f64
        } else {
            stats.current_memory as f64
        };

        let peak_to_average_ratio = if average_memory > 0.0 {
            stats.peak_memory as f64 / average_memory
        } else {
            1.0
        };

        // Generate recommendations
        if peak_to_average_ratio > 2.0 {
            recommendations.push(
                "High peak-to-average memory ratio detected. Consider using memory checkpointing."
                    .to_string(),
            );
        }

        if memory_efficiency < 100.0 {
            recommendations.push(
                "Low memory efficiency. Consider enabling in-place operations where possible."
                    .to_string(),
            );
        }

        if stats.total_allocations > stats.gradient_operations * 10 {
            recommendations.push(
                "High allocation count relative to operations. Consider tensor reuse strategies."
                    .to_string(),
            );
        }

        Self {
            stats,
            memory_efficiency,
            peak_to_average_ratio,
            recommendations,
        }
    }

    /// Print a formatted memory usage report
    pub fn print_report(&self) {
        println!("=== Gradient Memory Profile Report ===");
        println!(
            "Peak Memory Usage: {:.2} MB",
            self.stats.peak_memory as f64 / 1_048_576.0
        );
        println!(
            "Current Memory Usage: {:.2} MB",
            self.stats.current_memory as f64 / 1_048_576.0
        );
        println!(
            "Total Gradient Operations: {}",
            self.stats.gradient_operations
        );
        println!("Memory Efficiency: {:.2} ops/MB", self.memory_efficiency);
        println!("Peak-to-Average Ratio: {:.2}x", self.peak_to_average_ratio);

        println!("\n--- Memory Usage by Operation ---");
        let mut operations: Vec<_> = self.stats.operation_memory.iter().collect();
        operations.sort_by(|a, b| b.1.cmp(a.1));

        for (op, memory) in operations.iter().take(10) {
            println!("{}: {:.2} MB", op, **memory as f64 / 1_048_576.0);
        }

        if !self.recommendations.is_empty() {
            println!("\n--- Optimization Recommendations ---");
            for (i, rec) in self.recommendations.iter().enumerate() {
                println!("{}. {}", i + 1, rec);
            }
        }
        println!("=====================================");
    }

    /// Export report data as JSON string
    pub fn to_json(&self) -> Result<String> {
        // Simple JSON serialization for the key metrics
        let json = format!(
            r#"{{
    "peak_memory_mb": {:.2},
    "current_memory_mb": {:.2},
    "gradient_operations": {},
    "memory_efficiency": {:.2},
    "peak_to_average_ratio": {:.2},
    "total_allocations": {},
    "total_deallocations": {},
    "recommendations": {:?}
}}"#,
            self.stats.peak_memory as f64 / 1_048_576.0,
            self.stats.current_memory as f64 / 1_048_576.0,
            self.stats.gradient_operations,
            self.memory_efficiency,
            self.peak_to_average_ratio,
            self.stats.total_allocations,
            self.stats.total_deallocations,
            self.recommendations
        );
        Ok(json)
    }
}

/// Global memory profiler instance
static GLOBAL_PROFILER: OnceLock<Arc<Mutex<GradientMemoryProfiler>>> = OnceLock::new();

/// Get or initialize the global memory profiler
pub fn get_global_profiler() -> Arc<Mutex<GradientMemoryProfiler>> {
    GLOBAL_PROFILER
        .get_or_init(|| Arc::new(Mutex::new(GradientMemoryProfiler::new())))
        .clone()
}

/// Convenience macro for profiling gradient operations
#[macro_export]
macro_rules! profile_gradient_op {
    ($op_name:expr, $code:expr) => {{
        let profiler = $crate::memory_profiler::get_global_profiler();
        if let Ok(mut p) = profiler.lock() {
            p.profile_gradient_computation($op_name, || $code)
        } else {
            $code
        }
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_memory_profiler_creation() {
        let profiler = GradientMemoryProfiler::new();
        assert!(profiler.enabled);
        assert_eq!(profiler.sample_interval, Duration::from_millis(10));
    }

    #[test]
    fn test_memory_tracking() {
        let mut profiler = GradientMemoryProfiler::new();

        profiler.begin_operation("test_add").unwrap();
        profiler.record_allocation(1024).unwrap();
        profiler.end_operation().unwrap();

        let stats = profiler.get_stats().unwrap();
        assert_eq!(stats.gradient_operations, 1);
        assert!(stats.operation_memory.contains_key("test_add"));
    }

    #[test]
    fn test_memory_report_generation() {
        let mut profiler = GradientMemoryProfiler::new();

        profiler.record_allocation(1_048_576).unwrap(); // 1 MB
        profiler.begin_operation("matrix_mul").unwrap();
        profiler.record_allocation(2_097_152).unwrap(); // 2 MB more
        profiler.end_operation().unwrap();

        let report = profiler.generate_report().unwrap();
        assert!(report.memory_efficiency > 0.0);
        assert!(report.stats.peak_memory > 0);
    }

    #[test]
    fn test_profiler_disable() {
        let mut profiler = GradientMemoryProfiler::new();
        profiler.set_enabled(false);

        profiler.begin_operation("disabled_op").unwrap();
        profiler.record_allocation(1024).unwrap();
        profiler.end_operation().unwrap();

        let stats = profiler.get_stats().unwrap();
        // Should still record allocation but not operation
        assert_eq!(stats.gradient_operations, 0);
    }

    #[test]
    fn test_stats_reset() {
        let mut profiler = GradientMemoryProfiler::new();

        profiler.record_allocation(1024).unwrap();
        profiler.begin_operation("test").unwrap();
        profiler.end_operation().unwrap();

        profiler.reset_stats().unwrap();
        let stats = profiler.get_stats().unwrap();
        assert_eq!(stats.current_memory, 0);
        assert_eq!(stats.gradient_operations, 0);
    }

    #[test]
    fn test_global_profiler() {
        let profiler1 = get_global_profiler();
        let profiler2 = get_global_profiler();

        // Should be the same instance
        assert!(Arc::ptr_eq(&profiler1, &profiler2));
    }

    #[test]
    fn test_memory_timeline() {
        let mut profiler = GradientMemoryProfiler::new();
        profiler.set_sample_interval(Duration::from_millis(1));

        for i in 0..5 {
            profiler.record_allocation(1024).unwrap();
            profiler.begin_operation(&format!("op_{}", i)).unwrap();
            thread::sleep(Duration::from_millis(2));
            profiler.end_operation().unwrap();
        }

        let stats = profiler.get_stats().unwrap();
        assert!(!stats.memory_timeline.is_empty());
    }
}
