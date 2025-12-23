/// GPU Memory Allocation Tracing for TenfloweRS
///
/// This module provides comprehensive GPU memory allocation tracking and diagnostics
/// to help identify memory leaks, optimize memory usage, and provide detailed insights.
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Unique identifier for each GPU allocation
pub type AllocationId = u64;

/// Information about a single GPU memory allocation
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct AllocationInfo {
    /// Unique identifier for this allocation
    pub id: AllocationId,
    /// Size in bytes
    pub size: usize,
    /// Device ID
    pub device_id: usize,
    /// Timestamp when allocated
    #[cfg_attr(feature = "serialize", serde(skip, default = "Instant::now"))]
    pub allocated_at: Instant,
    /// Stack trace at allocation point (if enabled)
    pub stack_trace: Option<String>,
    /// Operation that triggered this allocation
    pub operation: String,
    /// Tensor shape (if applicable)
    pub shape: Option<Vec<usize>>,
    /// Data type name
    pub dtype: Option<String>,
    /// Whether this allocation is still active
    pub is_active: bool,
    /// When this allocation was freed (if freed)
    #[cfg_attr(feature = "serialize", serde(skip))]
    pub freed_at: Option<Instant>,
    /// Lifetime duration (if freed)
    #[cfg_attr(feature = "serialize", serde(skip))]
    pub lifetime: Option<Duration>,
    /// Custom metadata tags
    pub tags: HashMap<String, String>,
}

impl AllocationInfo {
    /// Create a new allocation info
    pub fn new(id: AllocationId, size: usize, device_id: usize, operation: String) -> Self {
        Self {
            id,
            size,
            device_id,
            allocated_at: Instant::now(),
            stack_trace: None,
            operation,
            shape: None,
            dtype: None,
            is_active: true,
            freed_at: None,
            lifetime: None,
            tags: HashMap::new(),
        }
    }

    /// Add shape information
    pub fn with_shape(mut self, shape: Vec<usize>) -> Self {
        self.shape = Some(shape);
        self
    }

    /// Add data type information
    pub fn with_dtype(mut self, dtype: String) -> Self {
        self.dtype = Some(dtype);
        self
    }

    /// Add a tag
    pub fn with_tag(mut self, key: String, value: String) -> Self {
        self.tags.insert(key, value);
        self
    }

    /// Mark this allocation as freed
    pub fn mark_freed(&mut self) {
        self.is_active = false;
        self.freed_at = Some(Instant::now());
        self.lifetime = Some(self.allocated_at.elapsed());
    }

    /// Get age of this allocation
    pub fn age(&self) -> Duration {
        if let Some(freed_at) = self.freed_at {
            freed_at.duration_since(self.allocated_at)
        } else {
            self.allocated_at.elapsed()
        }
    }
}

/// Statistics for GPU memory usage
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct MemoryStats {
    /// Total bytes currently allocated
    pub total_allocated: usize,
    /// Total bytes allocated over lifetime
    pub total_allocated_lifetime: usize,
    /// Total bytes freed over lifetime
    pub total_freed_lifetime: usize,
    /// Number of active allocations
    pub active_allocations: usize,
    /// Number of allocations over lifetime
    pub total_allocations_lifetime: u64,
    /// Number of frees over lifetime
    pub total_frees_lifetime: u64,
    /// Peak memory usage
    pub peak_usage: usize,
    /// Average allocation size
    pub average_allocation_size: usize,
    /// Largest allocation size
    pub largest_allocation: usize,
    /// Smallest allocation size
    pub smallest_allocation: Option<usize>,
}

/// Per-device memory statistics
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct DeviceMemoryStats {
    /// Device ID
    pub device_id: usize,
    /// Memory statistics
    pub stats: MemoryStats,
    /// Allocations by operation
    pub allocations_by_operation: HashMap<String, usize>,
    /// Memory by operation
    pub memory_by_operation: HashMap<String, usize>,
}

/// Memory allocation event
#[derive(Debug, Clone)]
pub enum MemoryEvent {
    /// Allocation event
    Allocated {
        id: AllocationId,
        size: usize,
        device_id: usize,
        operation: String,
    },
    /// Free event
    Freed {
        id: AllocationId,
        size: usize,
        device_id: usize,
        lifetime: Duration,
    },
    /// Out of memory event
    OutOfMemory {
        device_id: usize,
        requested_size: usize,
        available_size: usize,
    },
}

/// Configuration for memory tracing
#[derive(Debug, Clone)]
pub struct MemoryTracingConfig {
    /// Whether tracing is enabled
    pub enabled: bool,
    /// Capture stack traces
    pub capture_stack_traces: bool,
    /// Maximum number of allocations to track
    pub max_tracked_allocations: usize,
    /// Record detailed allocation history
    pub record_history: bool,
    /// Log memory events
    pub log_events: bool,
}

impl Default for MemoryTracingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            capture_stack_traces: false,
            max_tracked_allocations: 100_000,
            record_history: true,
            log_events: false,
        }
    }
}

/// GPU Memory Allocation Tracker
pub struct GpuMemoryTracker {
    /// Configuration
    config: MemoryTracingConfig,
    /// Next allocation ID
    next_id: AllocationId,
    /// Active allocations
    active_allocations: HashMap<AllocationId, AllocationInfo>,
    /// Historical allocations (freed)
    historical_allocations: Vec<AllocationInfo>,
    /// Memory events log
    events: Vec<MemoryEvent>,
    /// Per-device statistics
    device_stats: HashMap<usize, DeviceMemoryStats>,
    /// Global statistics
    global_stats: MemoryStats,
}

impl GpuMemoryTracker {
    /// Create a new GPU memory tracker with default configuration
    pub fn new() -> Self {
        Self::with_config(MemoryTracingConfig::default())
    }

    /// Create a new GPU memory tracker with custom configuration
    pub fn with_config(config: MemoryTracingConfig) -> Self {
        Self {
            config,
            next_id: 0,
            active_allocations: HashMap::new(),
            historical_allocations: Vec::new(),
            events: Vec::new(),
            device_stats: HashMap::new(),
            global_stats: MemoryStats::default(),
        }
    }

    /// Record a new allocation
    pub fn record_allocation(
        &mut self,
        size: usize,
        device_id: usize,
        operation: String,
    ) -> AllocationId {
        if !self.config.enabled {
            return 0;
        }

        let id = self.next_id;
        self.next_id += 1;

        let info = AllocationInfo::new(id, size, device_id, operation.clone());

        // Update global stats
        self.global_stats.total_allocated += size;
        self.global_stats.total_allocated_lifetime += size;
        self.global_stats.active_allocations += 1;
        self.global_stats.total_allocations_lifetime += 1;
        if self.global_stats.total_allocated > self.global_stats.peak_usage {
            self.global_stats.peak_usage = self.global_stats.total_allocated;
        }
        if size > self.global_stats.largest_allocation {
            self.global_stats.largest_allocation = size;
        }
        if self.global_stats.smallest_allocation.is_none()
            || size < self.global_stats.smallest_allocation.unwrap()
        {
            self.global_stats.smallest_allocation = Some(size);
        }

        // Update device stats
        let device_stats =
            self.device_stats
                .entry(device_id)
                .or_insert_with(|| DeviceMemoryStats {
                    device_id,
                    stats: MemoryStats::default(),
                    allocations_by_operation: HashMap::new(),
                    memory_by_operation: HashMap::new(),
                });

        device_stats.stats.total_allocated += size;
        device_stats.stats.total_allocated_lifetime += size;
        device_stats.stats.active_allocations += 1;
        device_stats.stats.total_allocations_lifetime += 1;
        *device_stats
            .allocations_by_operation
            .entry(operation.clone())
            .or_insert(0) += 1;
        *device_stats
            .memory_by_operation
            .entry(operation.clone())
            .or_insert(0) += size;

        // Record event
        if self.config.log_events {
            self.events.push(MemoryEvent::Allocated {
                id,
                size,
                device_id,
                operation,
            });
        }

        // Track allocation
        self.active_allocations.insert(id, info);

        id
    }

    /// Record a deallocation
    pub fn record_deallocation(&mut self, id: AllocationId) {
        if !self.config.enabled {
            return;
        }

        if let Some(mut info) = self.active_allocations.remove(&id) {
            let size = info.size;
            let device_id = info.device_id;
            info.mark_freed();

            // Update global stats
            self.global_stats.total_allocated -= size;
            self.global_stats.total_freed_lifetime += size;
            self.global_stats.active_allocations -= 1;
            self.global_stats.total_frees_lifetime += 1;

            // Update device stats
            if let Some(device_stats) = self.device_stats.get_mut(&device_id) {
                device_stats.stats.total_allocated -= size;
                device_stats.stats.total_freed_lifetime += size;
                device_stats.stats.active_allocations -= 1;
                device_stats.stats.total_frees_lifetime += 1;
            }

            // Record event
            if self.config.log_events {
                self.events.push(MemoryEvent::Freed {
                    id,
                    size,
                    device_id,
                    lifetime: info.lifetime.unwrap_or(Duration::ZERO),
                });
            }

            // Store in history if enabled
            if self.config.record_history {
                self.historical_allocations.push(info);

                // Limit history size
                if self.historical_allocations.len() > self.config.max_tracked_allocations {
                    self.historical_allocations
                        .drain(0..self.config.max_tracked_allocations / 10);
                }
            }
        }
    }

    /// Record an out-of-memory event
    pub fn record_oom(&mut self, device_id: usize, requested_size: usize, available_size: usize) {
        if self.config.log_events {
            self.events.push(MemoryEvent::OutOfMemory {
                device_id,
                requested_size,
                available_size,
            });
        }
    }

    /// Track a new allocation with optional metadata
    pub fn track_allocation(
        &mut self,
        size: usize,
        device_id: usize,
        operation: String,
        shape: Option<Vec<usize>>,
        dtype: Option<String>,
    ) -> AllocationId {
        let id = self.record_allocation(size, device_id, operation);

        // Add optional metadata
        if let Some(info) = self.active_allocations.get_mut(&id) {
            if let Some(s) = shape {
                info.shape = Some(s);
            }
            if let Some(dt) = dtype {
                info.dtype = Some(dt);
            }
        }

        id
    }

    /// Track a deallocation
    pub fn track_free(&mut self, id: AllocationId) {
        self.record_deallocation(id);
    }

    /// Get current memory usage
    pub fn current_usage(&self) -> usize {
        self.global_stats.total_allocated
    }

    /// Get peak memory usage
    pub fn peak_usage(&self) -> usize {
        self.global_stats.peak_usage
    }

    /// Get global statistics
    pub fn global_stats(&self) -> &MemoryStats {
        &self.global_stats
    }

    /// Get device statistics
    pub fn device_stats(&self, device_id: usize) -> Option<&DeviceMemoryStats> {
        self.device_stats.get(&device_id)
    }

    /// Get all device statistics
    pub fn all_device_stats(&self) -> &HashMap<usize, DeviceMemoryStats> {
        &self.device_stats
    }

    /// Get active allocations
    pub fn active_allocations(&self) -> &HashMap<AllocationId, AllocationInfo> {
        &self.active_allocations
    }

    /// Get memory events
    pub fn events(&self) -> &[MemoryEvent] {
        &self.events
    }

    /// Find potential memory leaks (allocations older than threshold)
    pub fn find_potential_leaks(&self, age_threshold: Duration) -> Vec<&AllocationInfo> {
        self.active_allocations
            .values()
            .filter(|info| info.age() > age_threshold)
            .collect()
    }

    /// Get memory usage by operation
    pub fn usage_by_operation(&self) -> HashMap<String, usize> {
        let mut result = HashMap::new();
        for info in self.active_allocations.values() {
            *result.entry(info.operation.clone()).or_insert(0) += info.size;
        }
        result
    }

    /// Generate a memory report
    pub fn generate_report(&self) -> MemoryReport {
        let mut allocations_by_size: Vec<_> = self.active_allocations.values().collect();
        allocations_by_size.sort_by(|a, b| b.size.cmp(&a.size));

        let top_allocations: Vec<_> = allocations_by_size.into_iter().take(10).cloned().collect();

        MemoryReport {
            global_stats: self.global_stats.clone(),
            device_stats: self.device_stats.clone(),
            top_allocations,
            usage_by_operation: self.usage_by_operation(),
            potential_leaks: self
                .find_potential_leaks(Duration::from_secs(300))
                .into_iter()
                .cloned()
                .collect(),
        }
    }

    /// Reset all statistics
    pub fn reset(&mut self) {
        self.active_allocations.clear();
        self.historical_allocations.clear();
        self.events.clear();
        self.device_stats.clear();
        self.global_stats = MemoryStats::default();
        self.next_id = 0;
    }
}

/// Memory usage report
#[derive(Debug, Clone)]
pub struct MemoryReport {
    /// Global statistics
    pub global_stats: MemoryStats,
    /// Per-device statistics
    pub device_stats: HashMap<usize, DeviceMemoryStats>,
    /// Top 10 largest allocations
    pub top_allocations: Vec<AllocationInfo>,
    /// Memory usage by operation type
    pub usage_by_operation: HashMap<String, usize>,
    /// Potential memory leaks
    pub potential_leaks: Vec<AllocationInfo>,
}

impl MemoryReport {
    /// Print a human-readable report
    pub fn print(&self) {
        println!("=== GPU Memory Usage Report ===");
        println!("\nGlobal Statistics:");
        println!(
            "  Current Allocation: {:.2} MB",
            self.global_stats.total_allocated as f64 / 1_048_576.0
        );
        println!(
            "  Peak Usage:         {:.2} MB",
            self.global_stats.peak_usage as f64 / 1_048_576.0
        );
        println!(
            "  Active Allocations: {}",
            self.global_stats.active_allocations
        );
        println!(
            "  Total Allocations:  {}",
            self.global_stats.total_allocations_lifetime
        );
        println!(
            "  Total Frees:        {}",
            self.global_stats.total_frees_lifetime
        );

        println!("\nTop 10 Allocations:");
        for (i, alloc) in self.top_allocations.iter().enumerate() {
            println!(
                "  {}: {:.2} MB - {} (age: {:.2}s)",
                i + 1,
                alloc.size as f64 / 1_048_576.0,
                alloc.operation,
                alloc.age().as_secs_f64()
            );
        }

        println!("\nMemory by Operation:");
        let mut ops: Vec<_> = self.usage_by_operation.iter().collect();
        ops.sort_by(|a, b| b.1.cmp(a.1));
        for (op, size) in ops.iter().take(10) {
            println!("  {}: {:.2} MB", op, **size as f64 / 1_048_576.0);
        }

        if !self.potential_leaks.is_empty() {
            println!("\n⚠️  Potential Memory Leaks:");
            for leak in &self.potential_leaks {
                println!(
                    "  {} - {:.2} MB (age: {:.2}s)",
                    leak.operation,
                    leak.size as f64 / 1_048_576.0,
                    leak.age().as_secs_f64()
                );
            }
        }

        println!("\n=============================");
    }
}

/// Global GPU memory tracker instance
lazy_static::lazy_static! {
    pub static ref GLOBAL_GPU_MEMORY_TRACKER: Arc<Mutex<GpuMemoryTracker>> = {
        Arc::new(Mutex::new(GpuMemoryTracker::new()))
    };
}

/// Convenience function to record an allocation
pub fn record_gpu_allocation(size: usize, device_id: usize, operation: String) -> AllocationId {
    GLOBAL_GPU_MEMORY_TRACKER
        .lock()
        .unwrap()
        .record_allocation(size, device_id, operation)
}

/// Convenience function to record a deallocation
pub fn record_gpu_deallocation(id: AllocationId) {
    GLOBAL_GPU_MEMORY_TRACKER
        .lock()
        .unwrap()
        .record_deallocation(id);
}

/// Convenience function to get current GPU memory usage
pub fn current_gpu_memory_usage() -> usize {
    GLOBAL_GPU_MEMORY_TRACKER.lock().unwrap().current_usage()
}

/// Convenience function to get peak GPU memory usage
pub fn peak_gpu_memory_usage() -> usize {
    GLOBAL_GPU_MEMORY_TRACKER.lock().unwrap().peak_usage()
}

/// Convenience function to generate memory report
pub fn generate_gpu_memory_report() -> MemoryReport {
    GLOBAL_GPU_MEMORY_TRACKER.lock().unwrap().generate_report()
}

/// Convenience function to print memory report
pub fn print_gpu_memory_report() {
    generate_gpu_memory_report().print();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocation_tracking() {
        let mut tracker = GpuMemoryTracker::new();

        let id1 = tracker.record_allocation(1024, 0, "test_op1".to_string());
        assert_eq!(tracker.current_usage(), 1024);
        assert_eq!(tracker.global_stats().active_allocations, 1);

        let id2 = tracker.record_allocation(2048, 0, "test_op2".to_string());
        assert_eq!(tracker.current_usage(), 3072);
        assert_eq!(tracker.global_stats().active_allocations, 2);

        tracker.record_deallocation(id1);
        assert_eq!(tracker.current_usage(), 2048);
        assert_eq!(tracker.global_stats().active_allocations, 1);

        tracker.record_deallocation(id2);
        assert_eq!(tracker.current_usage(), 0);
        assert_eq!(tracker.global_stats().active_allocations, 0);
    }

    #[test]
    fn test_peak_tracking() {
        let mut tracker = GpuMemoryTracker::new();

        let id1 = tracker.record_allocation(1024, 0, "test".to_string());
        let id2 = tracker.record_allocation(2048, 0, "test".to_string());
        assert_eq!(tracker.peak_usage(), 3072);

        tracker.record_deallocation(id1);
        tracker.record_deallocation(id2);
        assert_eq!(tracker.peak_usage(), 3072); // Peak should remain
        assert_eq!(tracker.current_usage(), 0);
    }

    #[test]
    fn test_usage_by_operation() {
        let mut tracker = GpuMemoryTracker::new();

        tracker.record_allocation(1024, 0, "op_a".to_string());
        tracker.record_allocation(2048, 0, "op_a".to_string());
        tracker.record_allocation(512, 0, "op_b".to_string());

        let usage = tracker.usage_by_operation();
        assert_eq!(usage.get("op_a"), Some(&3072));
        assert_eq!(usage.get("op_b"), Some(&512));
    }
}
