//! GPU Profiling Support
//!
//! This module provides GPU profiling capabilities for performance analysis
//! and optimization of GPU operations.

use crate::{Device, Result, TensorError};
use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc, Mutex,
};
use std::time::{Duration, Instant};

/// GPU profiling statistics
#[derive(Debug, Clone)]
pub struct ProfileStats {
    /// Total execution time
    pub total_time: Duration,
    /// Number of operations
    pub op_count: u64,
    /// Average time per operation
    pub avg_time: Duration,
    /// Peak memory usage (bytes)
    pub peak_memory: u64,
    /// Total GPU utilization percentage
    pub gpu_utilization: f32,
    /// Memory bandwidth utilization
    pub memory_bandwidth_utilization: f32,
}

/// GPU operation profile data
#[derive(Debug, Clone)]
pub struct OperationProfile {
    /// Operation name
    pub name: String,
    /// Execution time
    pub execution_time: Duration,
    /// Memory usage
    pub memory_usage: u64,
    /// Kernel occupancy percentage
    pub occupancy: f32,
    /// Device used
    pub device: Device,
    /// Timestamp
    pub timestamp: Instant,
}

/// GPU Profiler for tracking GPU operations
pub struct GpuProfiler {
    /// Whether profiling is enabled
    enabled: AtomicBool,
    /// Internal profiler state
    inner: Arc<Mutex<GpuProfilerInner>>,
    /// Operation counter
    op_counter: AtomicU64,
}

#[derive(Debug)]
struct GpuProfilerInner {
    /// Collected operation profiles
    operations: Vec<OperationProfile>,
    /// Memory usage tracking
    memory_usage: HashMap<Device, u64>,
    /// Start time of profiling session
    start_time: Option<Instant>,
    /// Peak memory usage
    peak_memory: u64,
    /// Current session statistics
    session_stats: HashMap<String, ProfileStats>,
}

impl GpuProfiler {
    /// Create a new GPU profiler
    pub fn new() -> Self {
        Self {
            enabled: AtomicBool::new(false),
            inner: Arc::new(Mutex::new(GpuProfilerInner {
                operations: Vec::new(),
                memory_usage: HashMap::new(),
                start_time: None,
                peak_memory: 0,
                session_stats: HashMap::new(),
            })),
            op_counter: AtomicU64::new(0),
        }
    }

    /// Enable GPU profiling
    pub fn enable(&self) {
        self.enabled.store(true, Ordering::Relaxed);
        let mut inner = self.inner.lock().unwrap();
        inner.start_time = Some(Instant::now());
        inner.operations.clear();
        inner.memory_usage.clear();
        inner.peak_memory = 0;
        inner.session_stats.clear();
    }

    /// Disable GPU profiling
    pub fn disable(&self) {
        self.enabled.store(false, Ordering::Relaxed);
    }

    /// Check if profiling is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    /// Record a GPU operation
    pub fn record_operation(
        &self,
        name: &str,
        device: Device,
        execution_time: Duration,
        memory_usage: u64,
    ) -> Result<()> {
        if !self.is_enabled() {
            return Ok(());
        }

        let mut inner = self.inner.lock().unwrap();

        // Estimate occupancy (simplified calculation)
        let occupancy = self.estimate_occupancy(execution_time, memory_usage);

        let profile = OperationProfile {
            name: name.to_string(),
            execution_time,
            memory_usage,
            occupancy,
            device,
            timestamp: Instant::now(),
        };

        inner.operations.push(profile);

        // Update memory tracking
        *inner.memory_usage.entry(device).or_insert(0) += memory_usage;
        if inner.memory_usage.values().sum::<u64>() > inner.peak_memory {
            inner.peak_memory = inner.memory_usage.values().sum::<u64>();
        }

        self.op_counter.fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    /// Get current profiling statistics
    pub fn get_stats(&self) -> Result<ProfileStats> {
        let inner = self.inner.lock().unwrap();

        if inner.operations.is_empty() {
            return Ok(ProfileStats {
                total_time: Duration::from_nanos(0),
                op_count: 0,
                avg_time: Duration::from_nanos(0),
                peak_memory: 0,
                gpu_utilization: 0.0,
                memory_bandwidth_utilization: 0.0,
            });
        }

        let total_time: Duration = inner.operations.iter().map(|op| op.execution_time).sum();

        let op_count = inner.operations.len() as u64;
        let avg_time = total_time / op_count as u32;

        // Estimate GPU utilization (simplified)
        let gpu_utilization = self.estimate_gpu_utilization(&inner.operations);
        let memory_bandwidth_utilization =
            self.estimate_memory_bandwidth_utilization(&inner.operations);

        Ok(ProfileStats {
            total_time,
            op_count,
            avg_time,
            peak_memory: inner.peak_memory,
            gpu_utilization,
            memory_bandwidth_utilization,
        })
    }

    /// Get all recorded operations
    pub fn get_operations(&self) -> Vec<OperationProfile> {
        let inner = self.inner.lock().unwrap();
        inner.operations.clone()
    }

    /// Generate a detailed profiling report
    pub fn generate_report(&self) -> Result<String> {
        let stats = self.get_stats()?;
        let operations = self.get_operations();

        let mut report = String::new();
        report.push_str("=== GPU Profiling Report ===\n\n");

        report.push_str(&format!("Total Operations: {}\n", stats.op_count));
        report.push_str(&format!("Total Execution Time: {:?}\n", stats.total_time));
        report.push_str(&format!(
            "Average Time per Operation: {:?}\n",
            stats.avg_time
        ));
        report.push_str(&format!("Peak Memory Usage: {} bytes\n", stats.peak_memory));
        report.push_str(&format!("GPU Utilization: {:.2}%\n", stats.gpu_utilization));
        report.push_str(&format!(
            "Memory Bandwidth Utilization: {:.2}%\n",
            stats.memory_bandwidth_utilization
        ));

        report.push_str("\n=== Operation Breakdown ===\n");

        // Group operations by name
        let mut op_groups: HashMap<String, Vec<&OperationProfile>> = HashMap::new();
        for op in &operations {
            op_groups.entry(op.name.clone()).or_default().push(op);
        }

        for (op_name, ops) in op_groups {
            let total_time: Duration = ops.iter().map(|op| op.execution_time).sum();
            let count = ops.len();
            let avg_time = total_time / count as u32;
            let avg_occupancy: f32 = ops.iter().map(|op| op.occupancy).sum::<f32>() / count as f32;

            report.push_str(&format!(
                "{}: {} calls, avg {:?}, avg occupancy {:.1}%\n",
                op_name, count, avg_time, avg_occupancy
            ));
        }

        Ok(report)
    }

    /// Clear all profiling data
    pub fn clear(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.operations.clear();
        inner.memory_usage.clear();
        inner.peak_memory = 0;
        inner.session_stats.clear();
        self.op_counter.store(0, Ordering::Relaxed);
    }

    /// Profile a GPU operation with automatic timing
    pub fn profile_operation<F, R>(&self, name: &str, device: Device, operation: F) -> Result<R>
    where
        F: FnOnce() -> Result<R>,
    {
        if !self.is_enabled() {
            return operation();
        }

        let start_memory = self.get_current_memory_usage(device)?;
        let start_time = Instant::now();

        let result = operation()?;

        let execution_time = start_time.elapsed();
        let end_memory = self.get_current_memory_usage(device)?;
        let memory_usage = end_memory.saturating_sub(start_memory);

        self.record_operation(name, device, execution_time, memory_usage)?;

        Ok(result)
    }

    // Helper methods

    fn estimate_occupancy(&self, execution_time: Duration, memory_usage: u64) -> f32 {
        // Simplified occupancy estimation
        // In a real implementation, this would query GPU-specific metrics
        let base_occupancy = 75.0; // Base occupancy percentage
        let time_factor = (execution_time.as_nanos() as f32 / 1_000_000.0).min(1.0);
        let memory_factor = (memory_usage as f32 / (1024.0 * 1024.0)).min(1.0);

        (base_occupancy * time_factor * memory_factor).min(100.0)
    }

    fn estimate_gpu_utilization(&self, operations: &[OperationProfile]) -> f32 {
        if operations.is_empty() {
            return 0.0;
        }

        // Simplified GPU utilization calculation
        let total_time: Duration = operations.iter().map(|op| op.execution_time).sum();
        let avg_occupancy: f32 =
            operations.iter().map(|op| op.occupancy).sum::<f32>() / operations.len() as f32;

        // Assume we're measuring over a 1-second window
        (total_time.as_secs_f32() * avg_occupancy / 100.0).min(1.0) * 100.0
    }

    fn estimate_memory_bandwidth_utilization(&self, operations: &[OperationProfile]) -> f32 {
        if operations.is_empty() {
            return 0.0;
        }

        // Simplified memory bandwidth utilization
        let total_memory: u64 = operations.iter().map(|op| op.memory_usage).sum();
        let total_time: Duration = operations.iter().map(|op| op.execution_time).sum();

        if total_time.as_secs_f32() == 0.0 {
            return 0.0;
        }

        // Estimate bandwidth usage (simplified)
        let bandwidth_usage_gb_s =
            (total_memory as f32 / (1024.0 * 1024.0 * 1024.0)) / total_time.as_secs_f32();

        // Assume peak GPU memory bandwidth is around 500 GB/s (simplified)
        let peak_bandwidth = 500.0;
        (bandwidth_usage_gb_s / peak_bandwidth * 100.0).min(100.0)
    }

    fn get_current_memory_usage(&self, device: Device) -> Result<u64> {
        // Use the global PerformanceMonitor to get actual memory usage
        let monitor = crate::memory::global_monitor();
        match device {
            #[cfg(feature = "gpu")]
            Device::Gpu(_) => {
                // Get GPU memory usage from PerformanceMonitor and internal tracking
                let inner = self.inner.lock().unwrap();
                let gpu_memory = inner.memory_usage.get(&device).copied().unwrap_or(0);
                let global_memory = monitor.get_current_memory() as u64;
                // Return GPU-specific memory or a portion of global memory for GPU device
                Ok(gpu_memory.max(global_memory / 2)) // Assume GPU uses significant portion
            }
            #[cfg(feature = "rocm")]
            Device::Rocm(_) => {
                // Get ROCM memory usage from PerformanceMonitor and internal tracking
                let inner = self.inner.lock().unwrap();
                let rocm_memory = inner.memory_usage.get(&device).copied().unwrap_or(0);
                let global_memory = monitor.get_current_memory() as u64;
                // Return ROCM-specific memory or a portion of global memory for ROCM device
                Ok(rocm_memory.max(global_memory / 2)) // Assume ROCM uses significant portion
            }
            Device::Cpu => {
                // Get CPU memory usage from PerformanceMonitor
                Ok(monitor.get_current_memory() as u64)
            }
        }
    }
}

impl Default for GpuProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Global GPU profiler instance
static GLOBAL_PROFILER: std::sync::OnceLock<GpuProfiler> = std::sync::OnceLock::new();

/// Get the global GPU profiler instance
pub fn global_profiler() -> &'static GpuProfiler {
    GLOBAL_PROFILER.get_or_init(GpuProfiler::default)
}

/// Convenience function to enable global GPU profiling
pub fn enable_gpu_profiling() {
    global_profiler().enable();
}

/// Convenience function to disable global GPU profiling
pub fn disable_gpu_profiling() {
    global_profiler().disable();
}

/// Convenience function to get global profiling stats
pub fn get_gpu_profiling_stats() -> Result<ProfileStats> {
    global_profiler().get_stats()
}

/// Convenience function to generate global profiling report
pub fn generate_gpu_profiling_report() -> Result<String> {
    global_profiler().generate_report()
}

/// Macro to profile a GPU operation
#[macro_export]
macro_rules! profile_gpu_op {
    ($name:expr, $device:expr, $op:expr) => {
        $crate::gpu_profiler::global_profiler().profile_operation($name, $device, || $op)
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_profiler_basic_functionality() {
        let profiler = GpuProfiler::new();

        // Initially disabled
        assert!(!profiler.is_enabled());

        // Enable profiling
        profiler.enable();
        assert!(profiler.is_enabled());

        // Record some operations
        let device = Device::Gpu(0);
        profiler
            .record_operation("matmul", device, Duration::from_millis(10), 1024)
            .unwrap();
        profiler
            .record_operation("conv2d", device, Duration::from_millis(5), 512)
            .unwrap();

        // Get stats
        let stats = profiler.get_stats().unwrap();
        assert_eq!(stats.op_count, 2);
        assert!(stats.total_time >= Duration::from_millis(15));
        assert_eq!(stats.peak_memory, 1536); // 1024 + 512

        // Disable profiling
        profiler.disable();
        assert!(!profiler.is_enabled());
    }

    #[test]
    fn test_profiler_report_generation() {
        let profiler = GpuProfiler::new();
        profiler.enable();

        let device = Device::Gpu(0);
        profiler
            .record_operation("test_op", device, Duration::from_millis(1), 100)
            .unwrap();

        let report = profiler.generate_report().unwrap();
        assert!(report.contains("GPU Profiling Report"));
        assert!(report.contains("test_op"));
        assert!(report.contains("Total Operations: 1"));
    }

    #[test]
    fn test_profile_operation_macro() {
        enable_gpu_profiling();

        let device = Device::Gpu(0);
        let result = profile_gpu_op!("test_macro", device, {
            thread::sleep(Duration::from_millis(1));
            Ok(42)
        });

        assert_eq!(result.unwrap(), 42);

        let stats = get_gpu_profiling_stats().unwrap();
        assert!(stats.op_count >= 1);

        disable_gpu_profiling();
    }

    #[test]
    fn test_profiler_clear() {
        let profiler = GpuProfiler::new();
        profiler.enable();

        let device = Device::Gpu(0);
        profiler
            .record_operation("test", device, Duration::from_millis(1), 100)
            .unwrap();

        let stats_before = profiler.get_stats().unwrap();
        assert_eq!(stats_before.op_count, 1);

        profiler.clear();

        let stats_after = profiler.get_stats().unwrap();
        assert_eq!(stats_after.op_count, 0);
    }
}
