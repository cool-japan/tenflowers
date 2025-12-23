/// GPU Memory Metrics Exposure
///
/// This module provides public APIs for monitoring and reporting GPU memory usage,
/// allocation patterns, and pool statistics. It complements the existing internal
/// diagnostics with user-facing APIs for production monitoring.

#[cfg(feature = "gpu")]
use crate::memory::{DiagnosticMemoryPool, MemoryPool, MemoryPoolStats};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// GPU memory usage snapshot
#[derive(Debug, Clone)]
pub struct GpuMemorySnapshot {
    /// Total bytes allocated across all pools
    pub total_allocated: usize,
    /// Total bytes in use (not free)
    pub total_used: usize,
    /// Number of active allocations
    pub allocation_count: usize,
    /// Peak memory usage recorded
    pub peak_usage: usize,
    /// Number of allocation failures
    pub allocation_failures: usize,
    /// Average allocation size
    pub avg_allocation_size: usize,
    /// Fragmentation ratio (0.0 = no fragmentation, 1.0 = high fragmentation)
    pub fragmentation_ratio: f32,
}

impl Default for GpuMemorySnapshot {
    fn default() -> Self {
        Self {
            total_allocated: 0,
            total_used: 0,
            allocation_count: 0,
            peak_usage: 0,
            allocation_failures: 0,
            avg_allocation_size: 0,
            fragmentation_ratio: 0.0,
        }
    }
}

/// GPU memory metrics collector
#[derive(Debug)]
pub struct GpuMemoryMetrics {
    total_allocations: AtomicU64,
    total_deallocations: AtomicU64,
    peak_memory: AtomicU64,
    current_memory: AtomicU64,
    allocation_failures: AtomicU64,
}

impl Default for GpuMemoryMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuMemoryMetrics {
    /// Create a new metrics collector
    pub fn new() -> Self {
        Self {
            total_allocations: AtomicU64::new(0),
            total_deallocations: AtomicU64::new(0),
            peak_memory: AtomicU64::new(0),
            current_memory: AtomicU64::new(0),
            allocation_failures: AtomicU64::new(0),
        }
    }

    /// Record an allocation
    pub fn record_allocation(&self, size: usize) {
        self.total_allocations.fetch_add(1, Ordering::Relaxed);
        let current = self
            .current_memory
            .fetch_add(size as u64, Ordering::Relaxed)
            + size as u64;

        // Update peak if necessary
        let mut peak = self.peak_memory.load(Ordering::Relaxed);
        while current > peak {
            match self.peak_memory.compare_exchange_weak(
                peak,
                current,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => peak = x,
            }
        }
    }

    /// Record a deallocation
    pub fn record_deallocation(&self, size: usize) {
        self.total_deallocations.fetch_add(1, Ordering::Relaxed);
        self.current_memory
            .fetch_sub(size as u64, Ordering::Relaxed);
    }

    /// Record an allocation failure
    pub fn record_failure(&self) {
        self.allocation_failures.fetch_add(1, Ordering::Relaxed);
    }

    /// Get current memory usage snapshot
    pub fn snapshot(&self) -> GpuMemorySnapshot {
        let total_allocs = self.total_allocations.load(Ordering::Relaxed);
        let total_deallocs = self.total_deallocations.load(Ordering::Relaxed);
        let current_memory = self.current_memory.load(Ordering::Relaxed) as usize;
        let peak_memory = self.peak_memory.load(Ordering::Relaxed) as usize;
        let allocation_count = (total_allocs.saturating_sub(total_deallocs)) as usize;

        let avg_allocation_size = if allocation_count > 0 {
            current_memory / allocation_count
        } else {
            0
        };

        // Simple fragmentation estimate based on allocation count vs memory usage
        let fragmentation_ratio = if current_memory > 0 && allocation_count > 0 {
            let ideal_allocations = (current_memory / 1024).max(1); // Assume 1KB ideal size
            (allocation_count as f32 / ideal_allocations as f32).min(1.0)
        } else {
            0.0
        };

        GpuMemorySnapshot {
            total_allocated: current_memory,
            total_used: current_memory,
            allocation_count,
            peak_usage: peak_memory,
            allocation_failures: self.allocation_failures.load(Ordering::Relaxed) as usize,
            avg_allocation_size,
            fragmentation_ratio,
        }
    }

    /// Reset all metrics (useful for testing/benchmarking)
    pub fn reset(&self) {
        self.total_allocations.store(0, Ordering::Relaxed);
        self.total_deallocations.store(0, Ordering::Relaxed);
        self.peak_memory.store(0, Ordering::Relaxed);
        self.current_memory.store(0, Ordering::Relaxed);
        self.allocation_failures.store(0, Ordering::Relaxed);
    }

    /// Get total number of allocations
    pub fn total_allocations(&self) -> u64 {
        self.total_allocations.load(Ordering::Relaxed)
    }

    /// Get total number of deallocations
    pub fn total_deallocations(&self) -> u64 {
        self.total_deallocations.load(Ordering::Relaxed)
    }

    /// Get current memory usage in bytes
    pub fn current_usage(&self) -> usize {
        self.current_memory.load(Ordering::Relaxed) as usize
    }

    /// Get peak memory usage in bytes
    pub fn peak_usage(&self) -> usize {
        self.peak_memory.load(Ordering::Relaxed) as usize
    }

    /// Get number of allocation failures
    pub fn allocation_failures(&self) -> usize {
        self.allocation_failures.load(Ordering::Relaxed) as usize
    }
}

/// Global GPU memory metrics instance
lazy_static::lazy_static! {
    pub static ref GPU_MEMORY_METRICS: Arc<GpuMemoryMetrics> = Arc::new(GpuMemoryMetrics::new());
}

/// Get current GPU memory usage
pub fn get_gpu_memory_usage() -> usize {
    GPU_MEMORY_METRICS.current_usage()
}

/// Get peak GPU memory usage
pub fn get_gpu_peak_memory() -> usize {
    GPU_MEMORY_METRICS.peak_usage()
}

/// Get GPU memory snapshot
pub fn get_gpu_memory_snapshot() -> GpuMemorySnapshot {
    GPU_MEMORY_METRICS.snapshot()
}

/// Reset GPU memory metrics (for testing/benchmarking)
pub fn reset_gpu_memory_metrics() {
    GPU_MEMORY_METRICS.reset();
}

/// GPU memory statistics report
#[derive(Debug, Clone)]
pub struct GpuMemoryReport {
    /// Current snapshot
    pub snapshot: GpuMemorySnapshot,
    /// Memory utilization percentage (0-100)
    pub utilization: f32,
    /// Allocation efficiency (deallocations / allocations)
    pub efficiency: f32,
    /// Recommendations for optimization
    pub recommendations: Vec<String>,
}

/// Generate a comprehensive GPU memory report
pub fn generate_memory_report() -> GpuMemoryReport {
    let snapshot = get_gpu_memory_snapshot();

    let total_ops =
        GPU_MEMORY_METRICS.total_allocations() + GPU_MEMORY_METRICS.total_deallocations();
    let efficiency = if total_ops > 0 {
        (GPU_MEMORY_METRICS.total_deallocations() as f32
            / GPU_MEMORY_METRICS.total_allocations() as f32)
            * 100.0
    } else {
        100.0
    };

    // Calculate utilization based on peak vs current
    let utilization = if snapshot.peak_usage > 0 {
        (snapshot.total_used as f32 / snapshot.peak_usage as f32) * 100.0
    } else {
        0.0
    };

    let mut recommendations = Vec::new();

    // Add recommendations based on metrics
    if snapshot.allocation_failures > 0 {
        recommendations.push(format!(
            "High allocation failure rate detected ({} failures). Consider increasing memory pool size.",
            snapshot.allocation_failures
        ));
    }

    if snapshot.fragmentation_ratio > 0.5 {
        recommendations.push(format!(
            "High memory fragmentation detected ({:.1}%). Consider enabling memory compaction.",
            snapshot.fragmentation_ratio * 100.0
        ));
    }

    if snapshot.allocation_count > 10000 {
        recommendations.push(format!(
            "Large number of active allocations ({}) detected. Consider batch processing to reduce overhead.",
            snapshot.allocation_count
        ));
    }

    if snapshot.avg_allocation_size < 1024 {
        recommendations.push(format!(
            "Small average allocation size ({} bytes). Consider allocation coalescing.",
            snapshot.avg_allocation_size
        ));
    }

    if efficiency < 80.0 {
        recommendations.push(format!(
            "Memory efficiency is low ({:.1}%). Check for memory leaks.",
            efficiency
        ));
    }

    GpuMemoryReport {
        snapshot,
        utilization,
        efficiency,
        recommendations,
    }
}

/// Print GPU memory report to stdout
pub fn print_memory_report() {
    let report = generate_memory_report();

    println!("\n=== GPU Memory Report ===");
    println!(
        "Total Allocated: {} bytes ({:.2} MB)",
        report.snapshot.total_allocated,
        report.snapshot.total_allocated as f32 / 1024.0 / 1024.0
    );
    println!(
        "Peak Usage: {} bytes ({:.2} MB)",
        report.snapshot.peak_usage,
        report.snapshot.peak_usage as f32 / 1024.0 / 1024.0
    );
    println!("Active Allocations: {}", report.snapshot.allocation_count);
    println!(
        "Average Allocation Size: {} bytes",
        report.snapshot.avg_allocation_size
    );
    println!(
        "Fragmentation: {:.1}%",
        report.snapshot.fragmentation_ratio * 100.0
    );
    println!("Utilization: {:.1}%", report.utilization);
    println!("Efficiency: {:.1}%", report.efficiency);

    if !report.recommendations.is_empty() {
        println!("\nRecommendations:");
        for (i, rec) in report.recommendations.iter().enumerate() {
            println!("  {}. {}", i + 1, rec);
        }
    }

    println!("========================\n");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_creation() {
        let metrics = GpuMemoryMetrics::new();
        assert_eq!(metrics.current_usage(), 0);
        assert_eq!(metrics.peak_usage(), 0);
        assert_eq!(metrics.total_allocations(), 0);
    }

    #[test]
    fn test_allocation_tracking() {
        let metrics = GpuMemoryMetrics::new();

        metrics.record_allocation(1024);
        assert_eq!(metrics.current_usage(), 1024);
        assert_eq!(metrics.peak_usage(), 1024);
        assert_eq!(metrics.total_allocations(), 1);

        metrics.record_allocation(2048);
        assert_eq!(metrics.current_usage(), 3072);
        assert_eq!(metrics.peak_usage(), 3072);
        assert_eq!(metrics.total_allocations(), 2);
    }

    #[test]
    fn test_deallocation_tracking() {
        let metrics = GpuMemoryMetrics::new();

        metrics.record_allocation(2048);
        metrics.record_deallocation(1024);

        assert_eq!(metrics.current_usage(), 1024);
        assert_eq!(metrics.peak_usage(), 2048);
        assert_eq!(metrics.total_deallocations(), 1);
    }

    #[test]
    fn test_snapshot() {
        let metrics = GpuMemoryMetrics::new();

        metrics.record_allocation(1024);
        metrics.record_allocation(2048);
        metrics.record_deallocation(512);

        let snapshot = metrics.snapshot();

        assert_eq!(snapshot.total_allocated, 2560); // 1024 + 2048 - 512
        assert_eq!(snapshot.allocation_count, 1); // 2 - 1
        assert_eq!(snapshot.peak_usage, 3072); // Peak before dealloc
    }

    #[test]
    fn test_failure_tracking() {
        let metrics = GpuMemoryMetrics::new();

        metrics.record_failure();
        metrics.record_failure();

        assert_eq!(metrics.allocation_failures(), 2);
    }

    #[test]
    fn test_reset() {
        let metrics = GpuMemoryMetrics::new();

        metrics.record_allocation(1024);
        metrics.record_failure();

        metrics.reset();

        assert_eq!(metrics.current_usage(), 0);
        assert_eq!(metrics.peak_usage(), 0);
        assert_eq!(metrics.total_allocations(), 0);
        assert_eq!(metrics.allocation_failures(), 0);
    }

    #[test]
    fn test_report_generation() {
        reset_gpu_memory_metrics();

        GPU_MEMORY_METRICS.record_allocation(1024);
        GPU_MEMORY_METRICS.record_allocation(2048);

        let report = generate_memory_report();

        assert!(report.snapshot.total_allocated > 0);
        assert!(report.utilization >= 0.0);
        assert!(report.efficiency >= 0.0);
    }
}
