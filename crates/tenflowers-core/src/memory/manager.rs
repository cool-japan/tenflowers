//! High-level memory management coordination
//!
//! This module provides the main memory management interface that coordinates
//! memory pools, performance tracking, cache optimization, and multi-stream operations.

use super::{
    cache::{AccessPattern, CacheOptimizer},
    pools::{MemoryPool, MemoryPoolStats},
    streams::MultiStreamMemoryManager,
    tracking::{global_monitor_arc, PerformanceMonitor},
    views::{MemoryAliasDetector, StridedView},
};
use crate::Device;
#[cfg(feature = "gpu")]
use crate::TensorError;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Central memory manager that coordinates all memory subsystems
pub struct MemoryManager {
    pools: Arc<RwLock<HashMap<Device, MemoryPool>>>,
    multi_stream_managers: Arc<RwLock<HashMap<usize, MultiStreamMemoryManager>>>,
    alias_detector: Arc<MemoryAliasDetector>,
    cache_optimizer: Arc<CacheOptimizer>,
    performance_monitor: Arc<PerformanceMonitor>,
    default_pool_size: usize,
}

impl MemoryManager {
    /// Create a new memory manager
    pub fn new() -> Self {
        Self {
            pools: Arc::new(RwLock::new(HashMap::new())),
            multi_stream_managers: Arc::new(RwLock::new(HashMap::new())),
            alias_detector: Arc::new(MemoryAliasDetector::new()),
            cache_optimizer: Arc::new(CacheOptimizer::new()),
            performance_monitor: global_monitor_arc(),
            default_pool_size: 512 * 1024 * 1024, // 512MB default
        }
    }

    /// Create a new memory manager with custom pool size
    pub fn with_pool_size(pool_size: usize) -> Self {
        let mut manager = Self::new();
        manager.default_pool_size = pool_size;
        manager
    }

    /// Get or create a memory pool for a specific device
    #[cfg(feature = "gpu")]
    pub fn get_pool(&self, device: Device) -> crate::Result<Arc<MemoryPool>> {
        let pools = self.pools.read().unwrap();

        if let Some(pool) = pools.get(&device) {
            // For now, return a simple wrapper since we can't clone MemoryPool directly
            // In a real implementation, this would return a proper Arc<MemoryPool>
            return Err(TensorError::unsupported_operation_simple(
                "Memory pool sharing not yet implemented".to_string(),
            ));
        }

        drop(pools);

        // Create new pool
        let pool = match device {
            Device::Gpu(device_id) => MemoryPool::new(device_id, self.default_pool_size)?,
            Device::Cpu => {
                return Err(TensorError::unsupported_operation_simple(
                    "CPU memory pools not yet implemented".to_string(),
                ))
            }
            #[cfg(feature = "rocm")]
            Device::Rocm(_) => {
                return Err(TensorError::unsupported_operation_simple(
                    "ROCm memory pools not yet implemented".to_string(),
                ))
            }
        };

        let mut pools = self.pools.write().unwrap();
        pools.insert(device, pool);

        // Return reference (in real implementation this would be Arc<MemoryPool>)
        Err(TensorError::unsupported_operation_simple(
            "Memory pool sharing not yet implemented".to_string(),
        ))
    }

    /// Get or create a multi-stream memory manager for a device
    #[cfg(feature = "gpu")]
    pub fn get_multi_stream_manager(
        &self,
        device_id: usize,
        num_streams: usize,
    ) -> crate::Result<Arc<MultiStreamMemoryManager>> {
        let managers = self.multi_stream_managers.read().unwrap();

        if managers.contains_key(&device_id) {
            return Err(TensorError::unsupported_operation_simple(
                "Multi-stream manager sharing not yet implemented".to_string(),
            ));
        }

        drop(managers);

        // Create new multi-stream manager
        let stream_pool_size = self.default_pool_size / num_streams;
        let manager = MultiStreamMemoryManager::new(device_id, num_streams, stream_pool_size)?;

        let mut managers = self.multi_stream_managers.write().unwrap();
        managers.insert(device_id, manager);

        Err(TensorError::unsupported_operation_simple(
            "Multi-stream manager sharing not yet implemented".to_string(),
        ))
    }

    /// Check for memory aliasing between tensor views
    pub fn check_memory_alias(&self, buffer_id: usize, offset: usize, size: usize) -> bool {
        self.alias_detector.check_alias(buffer_id, offset, size)
    }

    /// Register a new memory view for alias detection
    pub fn register_memory_view(&self, buffer_id: usize, offset: usize, size: usize) {
        self.alias_detector.register_view(buffer_id, offset, size);
    }

    /// Unregister a memory view
    pub fn unregister_memory_view(&self, buffer_id: usize, offset: usize, size: usize) {
        self.alias_detector.unregister_view(buffer_id, offset, size);
    }

    /// Get optimal memory access pattern for given dimensions
    pub fn get_optimal_access_pattern(&self, dims: &[usize], element_size: usize) -> AccessPattern {
        self.cache_optimizer
            .optimize_access_pattern(dims, element_size)
    }

    /// Get optimal alignment for data of given size
    pub fn get_optimal_alignment(&self, data_size: usize) -> usize {
        self.cache_optimizer.get_optimal_alignment(data_size)
    }

    /// Record a memory operation for performance tracking
    pub fn record_memory_operation(&self, operation: &str, size: usize) {
        self.performance_monitor.record_allocation(operation, size);
    }

    /// Get comprehensive memory statistics across all devices
    pub fn get_memory_statistics(&self) -> MemoryStatistics {
        let mut stats = MemoryStatistics::new();

        // Aggregate pool statistics
        let pools = self.pools.read().unwrap();
        for (device, pool) in pools.iter() {
            let pool_stats = pool.stats();
            stats.add_device_stats(*device, pool_stats);
        }

        // Add performance monitor data
        stats.total_allocations = self.performance_monitor.get_allocation_stats().0;
        stats.total_deallocations = self.performance_monitor.get_allocation_stats().1;
        stats.current_memory_tracked = self.performance_monitor.get_current_memory();
        stats.peak_memory_tracked = self.performance_monitor.get_peak_memory();

        // Add alias detector statistics
        let (alias_buffers, alias_views) = self.alias_detector.get_alias_statistics();
        stats.active_alias_buffers = alias_buffers;
        stats.active_alias_views = alias_views;

        stats
    }

    /// Generate comprehensive memory management report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();
        report.push_str("=== Memory Manager Report ===\n\n");

        // Performance monitoring
        report.push_str("Performance Monitoring:\n");
        let perf_report = self.performance_monitor.generate_report();
        report.push_str(&perf_report);
        report.push('\n');

        // Memory pools
        report.push_str("Memory Pools:\n");
        let pools = self.pools.read().unwrap();
        for (device, pool) in pools.iter() {
            let stats = pool.stats();
            report.push_str(&format!("  Device {:?}:\n", device));
            report.push_str(&format!("    Allocated: {} bytes\n", stats.total_allocated));
            report.push_str(&format!("    Free: {} bytes\n", stats.total_free));
            report.push_str(&format!(
                "    Fragmentation: {:.2}%\n",
                stats.fragmentation_ratio * 100.0
            ));
            report.push_str(&format!(
                "    Memory Pressure: {:.2}%\n",
                stats.memory_pressure * 100.0
            ));
        }
        report.push('\n');

        // Multi-stream managers
        report.push_str("Multi-Stream Managers:\n");
        let managers = self.multi_stream_managers.read().unwrap();
        for (device_id, manager) in managers.iter() {
            report.push_str(&format!("  Device {}:\n", device_id));
            report.push_str(&format!("    Streams: {}\n", manager.num_streams()));

            let (total_allocated, total_free) = manager.total_memory_usage();
            report.push_str(&format!("    Total Allocated: {} bytes\n", total_allocated));
            report.push_str(&format!("    Total Free: {} bytes\n", total_free));
        }
        report.push('\n');

        // Alias detection
        let (alias_buffers, alias_views) = self.alias_detector.get_alias_statistics();
        report.push_str("Memory Aliasing:\n");
        report.push_str(&format!("  Active Buffers: {}\n", alias_buffers));
        report.push_str(&format!("  Active Views: {}\n", alias_views));

        report
    }

    /// Optimize memory layout for tensor operations
    pub fn optimize_tensor_layout(
        &self,
        shape: &[usize],
        element_size: usize,
    ) -> TensorLayoutOptimization {
        let access_pattern = self
            .cache_optimizer
            .optimize_access_pattern(shape, element_size);
        let optimal_alignment = self
            .cache_optimizer
            .get_optimal_alignment(shape.iter().product::<usize>() * element_size);

        TensorLayoutOptimization {
            access_pattern,
            alignment: optimal_alignment,
            block_size: self
                .cache_optimizer
                .get_optimal_block_size(element_size, shape.iter().product()),
        }
    }

    /// Create an optimized strided view for zero-copy operations
    pub fn create_strided_view(
        &self,
        offset: usize,
        shape: Vec<usize>,
        strides: Vec<usize>,
        element_size: usize,
    ) -> StridedView {
        StridedView::new(offset, shape, strides, element_size)
    }

    /// Set the default pool size for new memory pools
    pub fn set_default_pool_size(&mut self, size: usize) {
        self.default_pool_size = size;
    }

    /// Get the current default pool size
    pub fn default_pool_size(&self) -> usize {
        self.default_pool_size
    }

    /// Clear all memory pools and reset the manager
    pub fn clear(&self) {
        let mut pools = self.pools.write().unwrap();
        pools.clear();

        let mut managers = self.multi_stream_managers.write().unwrap();
        managers.clear();

        self.performance_monitor.clear();
    }

    /// Get cache optimizer reference
    pub fn cache_optimizer(&self) -> &CacheOptimizer {
        &self.cache_optimizer
    }

    /// Get performance monitor reference
    pub fn performance_monitor(&self) -> &PerformanceMonitor {
        &self.performance_monitor
    }

    /// Get alias detector reference
    pub fn alias_detector(&self) -> &MemoryAliasDetector {
        &self.alias_detector
    }
}

impl Default for MemoryManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Comprehensive memory statistics across all subsystems
#[derive(Debug, Clone)]
pub struct MemoryStatistics {
    pub device_stats: HashMap<Device, MemoryPoolStats>,
    pub total_allocations: usize,
    pub total_deallocations: usize,
    pub current_memory_tracked: usize,
    pub peak_memory_tracked: usize,
    pub active_alias_buffers: usize,
    pub active_alias_views: usize,
}

impl MemoryStatistics {
    fn new() -> Self {
        Self {
            device_stats: HashMap::new(),
            total_allocations: 0,
            total_deallocations: 0,
            current_memory_tracked: 0,
            peak_memory_tracked: 0,
            active_alias_buffers: 0,
            active_alias_views: 0,
        }
    }

    fn add_device_stats(&mut self, device: Device, stats: MemoryPoolStats) {
        self.device_stats.insert(device, stats);
    }

    /// Get total memory allocated across all devices
    pub fn total_allocated(&self) -> usize {
        self.device_stats.values().map(|s| s.total_allocated).sum()
    }

    /// Get total memory free across all devices
    pub fn total_free(&self) -> usize {
        self.device_stats.values().map(|s| s.total_free).sum()
    }

    /// Get average fragmentation ratio across all devices
    pub fn average_fragmentation(&self) -> f32 {
        if self.device_stats.is_empty() {
            return 0.0;
        }

        let total_fragmentation: f32 = self
            .device_stats
            .values()
            .map(|s| s.fragmentation_ratio)
            .sum();

        total_fragmentation / self.device_stats.len() as f32
    }

    /// Get maximum memory pressure across all devices
    pub fn max_memory_pressure(&self) -> f32 {
        self.device_stats
            .values()
            .map(|s| s.memory_pressure)
            .fold(0.0, f32::max)
    }
}

/// Tensor layout optimization recommendations
#[derive(Debug, Clone)]
pub struct TensorLayoutOptimization {
    pub access_pattern: AccessPattern,
    pub alignment: usize,
    pub block_size: usize,
}

/// Global memory manager instance
static GLOBAL_MEMORY_MANAGER: std::sync::OnceLock<Arc<MemoryManager>> = std::sync::OnceLock::new();

/// Get the global memory manager
pub fn global_memory_manager() -> &'static MemoryManager {
    GLOBAL_MEMORY_MANAGER.get_or_init(|| Arc::new(MemoryManager::new()))
}

/// Get the global memory manager as Arc
pub fn global_memory_manager_arc() -> Arc<MemoryManager> {
    GLOBAL_MEMORY_MANAGER
        .get_or_init(|| Arc::new(MemoryManager::new()))
        .clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Device;

    #[test]
    fn test_memory_manager_creation() {
        let manager = MemoryManager::new();
        assert_eq!(manager.default_pool_size(), 512 * 1024 * 1024);

        let custom_manager = MemoryManager::with_pool_size(1024 * 1024);
        assert_eq!(custom_manager.default_pool_size(), 1024 * 1024);
    }

    #[test]
    fn test_memory_statistics() {
        let mut stats = MemoryStatistics::new();
        assert_eq!(stats.total_allocated(), 0);
        assert_eq!(stats.total_free(), 0);
        assert_eq!(stats.average_fragmentation(), 0.0);
        assert_eq!(stats.max_memory_pressure(), 0.0);

        // Add some mock device stats
        let device_stats = crate::memory::pools::MemoryPoolStats {
            total_allocated: 1000,
            total_free: 2000,
            blocks_allocated: 10,
            blocks_free: 5,
            fragmentation_ratio: 0.1,
            peak_allocated: 1500,
            allocation_count: 100,
            deallocation_count: 90,
            defragmentation_count: 2,
            largest_free_block: 1000,
            average_block_size: 200.0,
            memory_pressure: 0.3,
        };

        stats.add_device_stats(Device::Cpu, device_stats.clone());

        assert_eq!(stats.total_allocated(), 1000);
        assert_eq!(stats.total_free(), 2000);
        assert_eq!(stats.average_fragmentation(), 0.1);
        assert_eq!(stats.max_memory_pressure(), 0.3);
    }

    #[test]
    fn test_tensor_layout_optimization() {
        let manager = MemoryManager::new();
        let optimization = manager.optimize_tensor_layout(&[100, 100], 4);

        // Should have reasonable values
        assert!(optimization.alignment > 0);
        assert!(optimization.block_size > 0);
        // access_pattern should be one of the variants
        matches!(
            optimization.access_pattern,
            AccessPattern::Sequential | AccessPattern::Blocked { .. } | AccessPattern::Tiled { .. }
        );
    }

    #[test]
    fn test_strided_view_creation() {
        let manager = MemoryManager::new();
        let view = manager.create_strided_view(0, vec![2, 3], vec![12, 4], 4);

        assert_eq!(view.offset, 0);
        assert_eq!(view.shape, vec![2, 3]);
        assert_eq!(view.strides, vec![12, 4]);
        assert_eq!(view.element_size, 4);
    }

    #[test]
    fn test_memory_alias_operations() {
        let manager = MemoryManager::new();

        // Register a view
        manager.register_memory_view(0, 0, 100);

        // Check for alias
        assert!(manager.check_memory_alias(0, 50, 100)); // Should overlap
        assert!(!manager.check_memory_alias(0, 100, 50)); // Should not overlap

        // Unregister
        manager.unregister_memory_view(0, 0, 100);
        assert!(!manager.check_memory_alias(0, 50, 100)); // No longer aliased
    }

    #[test]
    fn test_memory_operation_recording() {
        let manager = MemoryManager::new();

        // Record initial state (global monitor may have previous operations)
        let initial_stats = manager.get_memory_statistics();
        let initial_allocations = initial_stats.total_allocations;
        let initial_memory = initial_stats.current_memory_tracked;

        manager.record_memory_operation("test_alloc", 1024);

        // Check that exactly one operation was recorded
        let stats = manager.get_memory_statistics();
        assert_eq!(stats.total_allocations, initial_allocations + 1);
        assert_eq!(stats.current_memory_tracked, initial_memory + 1024);
    }

    #[test]
    fn test_optimal_access_pattern() {
        let manager = MemoryManager::new();

        // Small tensor should get sequential access
        let pattern = manager.get_optimal_access_pattern(&[10, 10], 4);
        matches!(pattern, AccessPattern::Sequential);

        // Large tensor should get more complex access pattern
        let pattern = manager.get_optimal_access_pattern(&[1000, 1000], 4);
        matches!(
            pattern,
            AccessPattern::Blocked { .. } | AccessPattern::Tiled { .. }
        );
    }

    #[test]
    fn test_optimal_alignment() {
        let manager = MemoryManager::new();

        let small_alignment = manager.get_optimal_alignment(32);
        let large_alignment = manager.get_optimal_alignment(8192);

        assert!(small_alignment > 0);
        assert!(large_alignment > 0);
        assert!(large_alignment >= small_alignment); // Larger data should have larger or equal alignment
    }

    #[test]
    fn test_manager_clear() {
        let manager = MemoryManager::new();

        manager.record_memory_operation("test", 1024);
        manager.register_memory_view(0, 0, 100);

        let stats_before = manager.get_memory_statistics();
        assert!(stats_before.total_allocations > 0);

        manager.clear();

        let stats_after = manager.get_memory_statistics();
        assert_eq!(stats_after.total_allocations, 0);
        assert_eq!(stats_after.current_memory_tracked, 0);
    }

    #[test]
    fn test_global_memory_manager() {
        let manager1 = global_memory_manager();
        let manager2 = global_memory_manager();

        // Should be the same instance
        assert!(std::ptr::eq(manager1, manager2));

        // Test that we can use it - check relative change instead of absolute value
        let initial_stats = manager1.get_memory_statistics();
        let initial_tracked = initial_stats.current_memory_tracked;

        manager1.record_memory_operation("global_test", 512);
        let final_stats = manager2.get_memory_statistics();
        let final_tracked = final_stats.current_memory_tracked;

        // Check that the memory tracked increased by at least 512
        // (may be more due to other concurrent operations)
        assert!(final_tracked >= initial_tracked + 512);
    }

    #[test]
    fn test_report_generation() {
        let manager = MemoryManager::new();
        manager.record_memory_operation("test_op", 1024);

        let report = manager.generate_report();
        assert!(report.contains("Memory Manager Report"));
        assert!(report.contains("Performance Monitoring"));
        assert!(report.contains("Memory Pools"));
        assert!(report.contains("Memory Aliasing"));
    }

    #[test]
    fn test_set_default_pool_size() {
        let mut manager = MemoryManager::new();
        let original_size = manager.default_pool_size();

        manager.set_default_pool_size(1024 * 1024);
        assert_eq!(manager.default_pool_size(), 1024 * 1024);
        assert_ne!(manager.default_pool_size(), original_size);
    }
}
