//! Performance monitoring and allocation analytics
//!
//! This module provides comprehensive tracking of memory operations,
//! kernel performance, and system-wide memory usage statistics.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Kernel occupancy statistics for GPU performance analysis
#[derive(Debug, Clone)]
pub struct KernelOccupancyStats {
    pub kernel_name: String,
    pub workgroup_size: u32,
    pub workgroups_dispatched: u32,
    pub theoretical_occupancy: f32,
    pub achieved_occupancy: f32,
    pub efficiency_ratio: f32,
    pub memory_bandwidth_utilization: f32,
    pub arithmetic_intensity: f32,
}

/// Performance monitoring for operation timing and memory usage tracking
#[derive(Debug)]
pub struct PerformanceMonitor {
    inner: Arc<Mutex<PerformanceMonitorInner>>,
}

#[derive(Debug)]
struct PerformanceMonitorInner {
    operation_timings: HashMap<String, Vec<Duration>>,
    memory_usage: HashMap<String, usize>,
    total_allocations: usize,
    total_deallocations: usize,
    peak_memory: usize,
    current_memory: usize,
    kernel_occupancy: HashMap<String, Vec<KernelOccupancyStats>>,
}

impl Default for PerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl PerformanceMonitor {
    /// Create a new performance monitor
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(PerformanceMonitorInner {
                operation_timings: HashMap::new(),
                memory_usage: HashMap::new(),
                total_allocations: 0,
                total_deallocations: 0,
                peak_memory: 0,
                current_memory: 0,
                kernel_occupancy: HashMap::new(),
            })),
        }
    }

    /// Record the execution time of an operation
    pub fn record_operation_time(&self, operation: &str, duration: Duration) {
        if let Ok(mut inner) = self.inner.lock() {
            inner
                .operation_timings
                .entry(operation.to_string())
                .or_default()
                .push(duration);
        }
    }

    /// Record memory allocation
    pub fn record_allocation(&self, operation: &str, size: usize) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.memory_usage.insert(operation.to_string(), size);
            inner.total_allocations += 1;
            inner.current_memory += size;
            if inner.current_memory > inner.peak_memory {
                inner.peak_memory = inner.current_memory;
            }
        }
    }

    /// Record memory deallocation
    pub fn record_deallocation(&self, size: usize) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.total_deallocations += 1;
            inner.current_memory = inner.current_memory.saturating_sub(size);
        }
    }

    /// Get average execution time for an operation
    pub fn get_average_time(&self, operation: &str) -> Option<Duration> {
        if let Ok(inner) = self.inner.lock() {
            if let Some(times) = inner.operation_timings.get(operation) {
                if !times.is_empty() {
                    let total: Duration = times.iter().sum();
                    return Some(total / times.len() as u32);
                }
            }
        }
        None
    }

    /// Get all recorded operation times
    pub fn get_all_operation_times(&self) -> HashMap<String, Vec<Duration>> {
        if let Ok(inner) = self.inner.lock() {
            inner.operation_timings.clone()
        } else {
            HashMap::new()
        }
    }

    /// Get current memory usage
    pub fn get_current_memory(&self) -> usize {
        if let Ok(inner) = self.inner.lock() {
            inner.current_memory
        } else {
            0
        }
    }

    /// Get peak memory usage
    pub fn get_peak_memory(&self) -> usize {
        if let Ok(inner) = self.inner.lock() {
            inner.peak_memory
        } else {
            0
        }
    }

    /// Get memory allocation statistics
    pub fn get_allocation_stats(&self) -> (usize, usize) {
        if let Ok(inner) = self.inner.lock() {
            (inner.total_allocations, inner.total_deallocations)
        } else {
            (0, 0)
        }
    }

    /// Generate a performance report
    pub fn generate_report(&self) -> String {
        if let Ok(inner) = self.inner.lock() {
            let mut report = String::new();
            report.push_str("=== Performance Monitor Report ===\n\n");

            report.push_str("Memory Statistics:\n");
            report.push_str(&format!(
                "  Current Memory: {} bytes\n",
                inner.current_memory
            ));
            report.push_str(&format!("  Peak Memory: {} bytes\n", inner.peak_memory));
            report.push_str(&format!(
                "  Total Allocations: {}\n",
                inner.total_allocations
            ));
            report.push_str(&format!(
                "  Total Deallocations: {}\n",
                inner.total_deallocations
            ));
            report.push('\n');

            report.push_str("Operation Timings:\n");
            for (operation, times) in &inner.operation_timings {
                if !times.is_empty() {
                    let total: Duration = times.iter().sum();
                    let avg = total / times.len() as u32;
                    let min = times.iter().min().copied().unwrap_or_default();
                    let max = times.iter().max().copied().unwrap_or_default();

                    report.push_str(&format!("  {operation}:\n"));
                    report.push_str(&format!("    Count: {}\n", times.len()));
                    report.push_str(&format!("    Average: {avg:?}\n"));
                    report.push_str(&format!("    Min: {min:?}\n"));
                    report.push_str(&format!("    Max: {max:?}\n"));
                    report.push_str(&format!("    Total: {total:?}\n"));
                }
            }

            report
        } else {
            "Failed to generate report".to_string()
        }
    }

    /// Record kernel occupancy statistics
    pub fn record_kernel_occupancy(&self, stats: KernelOccupancyStats) {
        if let Ok(mut inner) = self.inner.lock() {
            inner
                .kernel_occupancy
                .entry(stats.kernel_name.clone())
                .or_default()
                .push(stats);
        }
    }

    /// Get kernel occupancy statistics for a specific kernel
    pub fn get_kernel_occupancy(&self, kernel_name: &str) -> Vec<KernelOccupancyStats> {
        if let Ok(inner) = self.inner.lock() {
            inner
                .kernel_occupancy
                .get(kernel_name)
                .cloned()
                .unwrap_or_default()
        } else {
            Vec::new()
        }
    }

    /// Get all kernel occupancy statistics
    pub fn get_all_kernel_occupancy(&self) -> HashMap<String, Vec<KernelOccupancyStats>> {
        if let Ok(inner) = self.inner.lock() {
            inner.kernel_occupancy.clone()
        } else {
            HashMap::new()
        }
    }

    /// Calculate average occupancy for a kernel
    pub fn get_average_kernel_occupancy(&self, kernel_name: &str) -> Option<f32> {
        if let Ok(inner) = self.inner.lock() {
            if let Some(stats) = inner.kernel_occupancy.get(kernel_name) {
                if !stats.is_empty() {
                    let total: f32 = stats.iter().map(|s| s.achieved_occupancy).sum();
                    return Some(total / stats.len() as f32);
                }
            }
        }
        None
    }

    /// Generate kernel occupancy analysis report
    pub fn generate_occupancy_report(&self) -> String {
        if let Ok(inner) = self.inner.lock() {
            let mut report = String::new();
            report.push_str("=== Kernel Occupancy Analysis ===\n\n");

            for (kernel_name, stats_vec) in &inner.kernel_occupancy {
                if !stats_vec.is_empty() {
                    let avg_occupancy: f32 =
                        stats_vec.iter().map(|s| s.achieved_occupancy).sum::<f32>()
                            / stats_vec.len() as f32;
                    let avg_efficiency: f32 =
                        stats_vec.iter().map(|s| s.efficiency_ratio).sum::<f32>()
                            / stats_vec.len() as f32;
                    let avg_bandwidth: f32 = stats_vec
                        .iter()
                        .map(|s| s.memory_bandwidth_utilization)
                        .sum::<f32>()
                        / stats_vec.len() as f32;
                    let avg_intensity: f32 = stats_vec
                        .iter()
                        .map(|s| s.arithmetic_intensity)
                        .sum::<f32>()
                        / stats_vec.len() as f32;

                    report.push_str(&format!("Kernel: {kernel_name}\n"));
                    report.push_str(&format!("  Invocations: {}\n", stats_vec.len()));
                    report.push_str(&format!("  Average Occupancy: {avg_occupancy:.2}%\n"));
                    report.push_str(&format!("  Average Efficiency: {avg_efficiency:.2}%\n"));
                    report.push_str(&format!(
                        "  Average Bandwidth Utilization: {avg_bandwidth:.2}%\n"
                    ));
                    report.push_str(&format!(
                        "  Average Arithmetic Intensity: {avg_intensity:.2}\n"
                    ));

                    // Performance recommendations
                    if avg_occupancy < 50.0 {
                        report.push_str(
                            "  ⚠️  Low occupancy detected. Consider increasing workgroup size.\n",
                        );
                    }
                    if avg_efficiency < 70.0 {
                        report.push_str(
                            "  ⚠️  Low efficiency. Check for thread divergence or memory issues.\n",
                        );
                    }
                    if avg_bandwidth < 60.0 {
                        report.push_str("  ⚠️  Low memory bandwidth utilization. Consider memory access optimization.\n");
                    }

                    report.push('\n');
                }
            }

            report
        } else {
            "Failed to generate occupancy report".to_string()
        }
    }

    /// Clear all recorded statistics
    pub fn clear(&self) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.operation_timings.clear();
            inner.memory_usage.clear();
            inner.total_allocations = 0;
            inner.total_deallocations = 0;
            inner.peak_memory = 0;
            inner.current_memory = 0;
            inner.kernel_occupancy.clear();
        }
    }
}

/// A timer for measuring operation execution time
pub struct OperationTimer {
    operation: String,
    start: Instant,
    monitor: Arc<PerformanceMonitor>,
}

impl OperationTimer {
    /// Create a new operation timer
    pub fn new(operation: String, monitor: Arc<PerformanceMonitor>) -> Self {
        Self {
            operation,
            start: Instant::now(),
            monitor,
        }
    }
}

impl Drop for OperationTimer {
    fn drop(&mut self) {
        let duration = self.start.elapsed();
        self.monitor
            .record_operation_time(&self.operation, duration);
    }
}

/// Global performance monitor instance
static GLOBAL_MONITOR: std::sync::OnceLock<Arc<PerformanceMonitor>> = std::sync::OnceLock::new();

/// Get the global performance monitor
pub fn global_monitor() -> &'static PerformanceMonitor {
    GLOBAL_MONITOR.get_or_init(|| Arc::new(PerformanceMonitor::new()))
}

/// Get the global performance monitor as Arc
pub fn global_monitor_arc() -> Arc<PerformanceMonitor> {
    GLOBAL_MONITOR
        .get_or_init(|| Arc::new(PerformanceMonitor::new()))
        .clone()
}

/// Macro for easily timing operations
#[macro_export]
macro_rules! time_operation {
    ($name:expr, $code:block) => {{
        let monitor = $crate::memory::tracking::global_monitor_arc();
        let _timer = $crate::memory::tracking::OperationTimer::new($name.to_string(), monitor);
        $code
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_performance_monitor() {
        let monitor = PerformanceMonitor::new();

        // Test operation timing
        monitor.record_operation_time("test_op", Duration::from_millis(100));
        monitor.record_operation_time("test_op", Duration::from_millis(200));

        let avg_time = monitor.get_average_time("test_op").unwrap();
        assert_eq!(avg_time, Duration::from_millis(150));

        // Test memory tracking
        monitor.record_allocation("tensor_alloc", 1024);
        assert_eq!(monitor.get_current_memory(), 1024);
        assert_eq!(monitor.get_peak_memory(), 1024);

        monitor.record_allocation("another_alloc", 512);
        assert_eq!(monitor.get_current_memory(), 1536);
        assert_eq!(monitor.get_peak_memory(), 1536);

        monitor.record_deallocation(512);
        assert_eq!(monitor.get_current_memory(), 1024);
        assert_eq!(monitor.get_peak_memory(), 1536); // Peak remains

        let (allocs, deallocs) = monitor.get_allocation_stats();
        assert_eq!(allocs, 2);
        assert_eq!(deallocs, 1);
    }

    #[test]
    fn test_operation_timer() {
        let monitor = Arc::new(PerformanceMonitor::new());

        {
            let _timer = OperationTimer::new("sleep_test".to_string(), monitor.clone());
            thread::sleep(Duration::from_millis(10));
        }

        let avg_time = monitor.get_average_time("sleep_test").unwrap();
        assert!(avg_time >= Duration::from_millis(9)); // Allow some variance
    }

    #[test]
    fn test_report_generation() {
        let monitor = PerformanceMonitor::new();
        monitor.record_operation_time("op1", Duration::from_millis(100));
        monitor.record_allocation("alloc1", 1024);

        let report = monitor.generate_report();
        assert!(report.contains("Performance Monitor Report"));
        assert!(report.contains("Current Memory: 1024 bytes"));
        assert!(report.contains("op1:"));
    }

    #[test]
    fn test_global_monitor() {
        let monitor1 = global_monitor();
        let monitor2 = global_monitor();

        // Should be the same instance
        assert!(std::ptr::eq(monitor1, monitor2));

        // Test that we can use it - use relative check for test isolation
        let initial_memory = monitor1.get_current_memory();

        monitor1.record_allocation("global_test", 512);
        let final_memory = monitor2.get_current_memory();

        // Check that memory increased by exactly 512
        assert_eq!(final_memory - initial_memory, 512);
    }

    #[test]
    fn test_kernel_occupancy() {
        let monitor = PerformanceMonitor::new();

        let stats = KernelOccupancyStats {
            kernel_name: "test_kernel".to_string(),
            workgroup_size: 256,
            workgroups_dispatched: 100,
            theoretical_occupancy: 100.0,
            achieved_occupancy: 85.0,
            efficiency_ratio: 90.0,
            memory_bandwidth_utilization: 75.0,
            arithmetic_intensity: 2.5,
        };

        monitor.record_kernel_occupancy(stats);

        let avg_occupancy = monitor.get_average_kernel_occupancy("test_kernel").unwrap();
        assert_eq!(avg_occupancy, 85.0);

        let occupancy_report = monitor.generate_occupancy_report();
        assert!(occupancy_report.contains("Kernel Occupancy Analysis"));
        assert!(occupancy_report.contains("test_kernel"));
    }

    #[test]
    fn test_clear_statistics() {
        let monitor = PerformanceMonitor::new();

        monitor.record_operation_time("op", Duration::from_millis(100));
        monitor.record_allocation("alloc", 1024);

        assert_eq!(monitor.get_current_memory(), 1024);
        assert!(monitor.get_average_time("op").is_some());

        monitor.clear();

        assert_eq!(monitor.get_current_memory(), 0);
        assert!(monitor.get_average_time("op").is_none());
    }
}
