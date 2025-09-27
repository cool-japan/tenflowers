//! Memory management infrastructure for TenfloweRS
//!
//! This module has been refactored into a modular structure for better maintainability.
//! All public APIs remain the same for backward compatibility.
//!
//! The memory management framework is now organized into specialized modules:
//! - Memory pool management with reference counting and defragmentation
//! - Performance monitoring and allocation analytics
//! - Zero-copy tensor operations and memory views
//! - Multi-stream memory management for concurrent operations
//! - Cache optimization utilities for improved performance
//! - High-level memory management coordination

pub mod cache;
pub mod manager;
pub mod pools;
pub mod streams;
pub mod tracking;
pub mod ultra_cache_optimizer;
pub mod ultra_efficient_pool_simple;
pub mod unified_optimizer;
pub mod views;

// Re-export all public types for backward compatibility

// Memory pool management
pub use pools::{
    align_size, AllocationTracker, BufferView, MemoryPool, MemoryPoolStats, MemoryPressureLevel,
    PooledBuffer,
};

// Performance monitoring and tracking
pub use tracking::{
    global_monitor, global_monitor_arc, KernelOccupancyStats, OperationTimer, PerformanceMonitor,
};

// Zero-copy operations and memory views
pub use views::{compute_strides, MemoryAliasDetector, StridedView};

// Multi-stream memory management
pub use streams::MultiStreamMemoryManager;

// Cache optimization utilities
pub use cache::{
    align_to_cache_line, global_cache_optimizer, is_cache_aligned, AccessPattern, CacheOptimizer,
    MatrixLayoutOptimizer, PrefetchOptimizer,
};

// High-level memory management
pub use manager::{
    global_memory_manager, global_memory_manager_arc, MemoryManager, MemoryStatistics,
    TensorLayoutOptimization,
};

// Ultra-efficient memory pool (SciRS2-Core powered)
pub use ultra_efficient_pool_simple::{
    global_memory_pool, profiling as ultra_profiling, MemoryStats, PoolConfig,
    UltraEfficientBuffer, UltraEfficientMemoryPool,
};

// Ultra-advanced cache optimization
pub use ultra_cache_optimizer::{
    global_cache_optimizer as global_ultra_cache_optimizer, AccessPatternAnalysis,
    CacheOptimizationStatistics, CacheOptimizerConfig, MemoryOptimizationResult, NumaTopology,
    OptimizationPriority, OptimizationType, PerformanceImpact, UltraCacheOptimizer,
};

// Unified ultra-performance optimization
pub use unified_optimizer::{
    global_unified_optimizer, OperationPerformanceProfile, OptimizationStrategy,
    PerformanceCharacteristics, UnifiedOptimizationEngine, UnifiedOptimizationResult,
    UnifiedOptimizationStatistics, UnifiedOptimizerConfig,
};

// Re-export the time_operation macro (exported at crate root due to #[macro_export])
pub use crate::time_operation;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backward_compatibility_imports() {
        // Test that all major types can be imported directly from the memory module

        // Memory pool types
        let _pressure = MemoryPressureLevel::Low;
        let _stats = MemoryPoolStats {
            total_allocated: 0,
            total_free: 0,
            blocks_allocated: 0,
            blocks_free: 0,
            fragmentation_ratio: 0.0,
            peak_allocated: 0,
            allocation_count: 0,
            deallocation_count: 0,
            defragmentation_count: 0,
            largest_free_block: 0,
            average_block_size: 0.0,
            memory_pressure: 0.0,
        };

        // Performance monitoring
        let _monitor = PerformanceMonitor::new();
        let _global_monitor = global_monitor();

        // Zero-copy operations
        let _view = StridedView::new(0, vec![2, 3], vec![12, 4], 4);
        let _detector = MemoryAliasDetector::new();

        // Cache optimization
        let _optimizer = CacheOptimizer::new();
        let _pattern = AccessPattern::Sequential;

        // High-level management
        let _manager = MemoryManager::new();
        let _global_manager = global_memory_manager();
    }

    #[test]
    fn test_utility_functions() {
        // Test utility functions are accessible
        let aligned = align_size(13, 8);
        assert_eq!(aligned, 16);

        let strides = compute_strides(&[2, 3, 4], 4);
        assert_eq!(strides, vec![48, 16, 4]);

        let cache_aligned = align_to_cache_line(50);
        assert_eq!(cache_aligned, 64);

        let ptr = 0x1000 as *const u8;
        assert!(is_cache_aligned(ptr));
    }

    #[test]
    fn test_global_instances() {
        // Test that global instances work correctly
        let monitor1 = global_monitor();
        let monitor2 = global_monitor();
        assert!(std::ptr::eq(monitor1, monitor2));

        let manager1 = global_memory_manager();
        let manager2 = global_memory_manager();
        assert!(std::ptr::eq(manager1, manager2));

        let cache_opt1 = global_cache_optimizer();
        let cache_opt2 = global_cache_optimizer();
        assert!(std::ptr::eq(cache_opt1, cache_opt2));
    }

    #[test]
    fn test_strided_view_operations() {
        // Test strided view functionality
        let view = StridedView::new(0, vec![2, 3, 4], vec![48, 16, 4], 4);

        // Test transpose
        let transposed = view.transpose(&[2, 0, 1]).unwrap();
        assert_eq!(transposed.shape, vec![4, 2, 3]);

        // Test reshape
        let reshaped = view.reshape(&[6, 4]).unwrap();
        assert_eq!(reshaped.shape, vec![6, 4]);

        // Test slice
        let view_2d = StridedView::new(0, vec![4, 4], vec![16, 4], 4);
        let sliced = view_2d.slice(&[(1, 3), (0, 2)]).unwrap();
        assert_eq!(sliced.shape, vec![2, 2]);
    }

    #[test]
    fn test_memory_aliasing() {
        // Test memory alias detection
        let detector = MemoryAliasDetector::new();

        detector.register_view(0, 0, 100);
        assert!(detector.check_alias(0, 50, 100)); // Overlaps
        assert!(!detector.check_alias(0, 100, 50)); // No overlap

        detector.unregister_view(0, 0, 100);
        assert!(!detector.check_alias(0, 50, 100)); // No longer aliased
    }

    #[test]
    fn test_performance_monitoring() {
        use std::time::Duration;

        let monitor = PerformanceMonitor::new();

        // Test operation timing
        monitor.record_operation_time("test_op", Duration::from_millis(100));
        monitor.record_operation_time("test_op", Duration::from_millis(200));

        let avg_time = monitor.get_average_time("test_op").unwrap();
        assert_eq!(avg_time, Duration::from_millis(150));

        // Test memory tracking
        monitor.record_allocation("tensor_alloc", 1024);
        assert_eq!(monitor.get_current_memory(), 1024);

        monitor.record_deallocation(512);
        assert_eq!(monitor.get_current_memory(), 512);
    }

    #[test]
    fn test_cache_optimization() {
        let optimizer = CacheOptimizer::new();

        // Test alignment calculation
        let alignment = optimizer.get_optimal_alignment(1024);
        assert!(alignment >= 1024);

        // Test access pattern optimization
        let pattern = optimizer.optimize_access_pattern(&[100, 100], 4);
        matches!(
            pattern,
            AccessPattern::Sequential | AccessPattern::Blocked { .. } | AccessPattern::Tiled { .. }
        );

        // Test block size calculation
        let block_size = optimizer.get_optimal_block_size(4, 1000);
        assert!(block_size >= 64);
        assert!(block_size <= 1000);
    }

    #[test]
    fn test_memory_manager_integration() {
        let manager = MemoryManager::new();

        // Test tensor layout optimization
        let optimization = manager.optimize_tensor_layout(&[100, 100], 4);
        assert!(optimization.alignment > 0);
        assert!(optimization.block_size > 0);

        // Test memory operation recording - use relative check for test isolation
        let initial_stats = manager.get_memory_statistics();
        let initial_tracked = initial_stats.current_memory_tracked;

        manager.record_memory_operation("test_alloc", 1024);
        let final_stats = manager.get_memory_statistics();
        let final_tracked = final_stats.current_memory_tracked;

        // Check that memory tracked increased by exactly 1024
        assert_eq!(final_tracked - initial_tracked, 1024);

        // Test alias operations
        manager.register_memory_view(0, 0, 100);
        assert!(manager.check_memory_alias(0, 50, 100));
        manager.unregister_memory_view(0, 0, 100);
        assert!(!manager.check_memory_alias(0, 50, 100));
    }

    #[test]
    fn test_matrix_layout_optimization() {
        let optimizer = MatrixLayoutOptimizer::new();

        // Test blocking transformation
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]; // 3x4 matrix
        let blocked = optimizer.to_blocked_layout(&data, 3, 4, 2);
        assert_eq!(blocked.len(), data.len());

        // Test reverse transformation
        let restored = optimizer.from_blocked_layout(&blocked, 3, 4, 2);
        assert_eq!(restored, data);
    }

    #[test]
    fn test_prefetch_optimization() {
        let optimizer = PrefetchOptimizer::new();

        // Test prefetch decision
        assert!(optimizer.should_prefetch(8, 2 * 1024 * 1024)); // Large data, large stride
        assert!(!optimizer.should_prefetch(2, 1024)); // Small data, small stride

        // Test prefetch methods (they don't return values, just ensure they don't panic)
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        optimizer.prefetch_sequential(&data, 0);
        optimizer.prefetch_strided(&data, 0, 2);
    }

    #[test]
    fn test_kernel_occupancy_tracking() {
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
    fn test_comprehensive_report_generation() {
        let manager = MemoryManager::new();
        manager.record_memory_operation("test_op", 1024);

        let report = manager.generate_report();
        assert!(report.contains("Memory Manager Report"));
        assert!(report.contains("Performance Monitoring"));
        assert!(report.contains("Memory Pools"));
        assert!(report.contains("Memory Aliasing"));
    }
}
