//! Ultra-Efficient Memory Pool Management with SciRS2-Core Integration
//!
//! This module provides state-of-the-art memory management for TenfloweRS,
//! leveraging SciRS2-Core's memory efficiency features for maximum performance
//! and minimal memory fragmentation. Designed for ultra-high-performance
//! deep learning workloads with humility and respect for system resources.

use crate::{Result, TensorError};
use scirs2_core::memory::{BufferPool, GlobalBufferPool, ChunkProcessor};
use scirs2_core::memory_efficient::{MemoryMappedArray, LazyArray, ChunkedArray};
use scirs2_core::memory_efficient::{ZeroCopyOps, AdaptiveChunking, DiskBackedArray};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

/// Ultra-efficient memory pool manager with adaptive allocation strategies
pub struct UltraEfficientMemoryPool {
    /// Primary buffer pools by element type and size class
    pools: RwLock<HashMap<PoolKey, Arc<BufferPool>>>,

    /// Global buffer pool for cross-operation sharing
    global_pool: Arc<GlobalBufferPool>,

    /// Adaptive chunking processor for large tensors
    chunk_processor: Arc<ChunkProcessor>,

    /// Memory usage statistics and optimization metrics
    stats: Arc<Mutex<MemoryStats>>,

    /// Configuration for pool behavior
    config: PoolConfig,
}

/// Memory pool configuration for maximum efficiency
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Maximum memory per pool (in bytes)
    pub max_pool_size: usize,

    /// Minimum allocation size for small objects
    pub min_allocation_size: usize,

    /// Maximum allocation size before using disk backing
    pub max_memory_allocation: usize,

    /// Enable zero-copy optimizations
    pub enable_zero_copy: bool,

    /// Enable adaptive chunking for large tensors
    pub enable_adaptive_chunking: bool,

    /// Memory pressure threshold (0.0 - 1.0)
    pub memory_pressure_threshold: f64,

    /// Enable aggressive memory optimization under pressure
    pub aggressive_optimization: bool,

    /// Background cleanup interval
    pub cleanup_interval: Duration,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            max_pool_size: 1_073_741_824, // 1GB per pool
            min_allocation_size: 4096,     // 4KB minimum
            max_memory_allocation: 536_870_912, // 512MB before disk backing
            enable_zero_copy: true,
            enable_adaptive_chunking: true,
            memory_pressure_threshold: 0.85, // Use 85% before optimization
            aggressive_optimization: true,
            cleanup_interval: Duration::from_secs(30),
        }
    }
}

/// Key for identifying buffer pools
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct PoolKey {
    element_size: usize,
    size_class: SizeClass,
    alignment: usize,
}

/// Size classes for efficient memory allocation
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
enum SizeClass {
    /// Small allocations (< 64KB)
    Small,
    /// Medium allocations (64KB - 16MB)
    Medium,
    /// Large allocations (16MB - 512MB)
    Large,
    /// Huge allocations (> 512MB) - use disk backing
    Huge,
}

/// Memory usage statistics for performance monitoring
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    /// Total bytes allocated
    pub total_allocated: usize,

    /// Total bytes freed
    pub total_freed: usize,

    /// Current memory usage
    pub current_usage: usize,

    /// Peak memory usage
    pub peak_usage: usize,

    /// Number of allocations
    pub allocation_count: u64,

    /// Number of deallocations
    pub deallocation_count: u64,

    /// Number of cache hits
    pub cache_hits: u64,

    /// Number of cache misses
    pub cache_misses: u64,

    /// Memory fragmentation ratio
    pub fragmentation_ratio: f64,

    /// Zero-copy operations count
    pub zero_copy_operations: u64,

    /// Adaptive chunking operations
    pub adaptive_chunking_ops: u64,

    /// Disk-backed operations
    pub disk_backed_ops: u64,

    /// Last cleanup time
    pub last_cleanup: Option<Instant>,
}

impl UltraEfficientMemoryPool {
    /// Create a new ultra-efficient memory pool with SciRS2-Core integration
    pub fn new(config: PoolConfig) -> Result<Self> {
        let global_pool = Arc::new(GlobalBufferPool::new(config.max_pool_size * 4)?);
        let chunk_processor = Arc::new(ChunkProcessor::new(
            config.max_memory_allocation,
            config.enable_adaptive_chunking,
        )?);

        Ok(Self {
            pools: RwLock::new(HashMap::new()),
            global_pool,
            chunk_processor,
            stats: Arc::new(Mutex::new(MemoryStats::default())),
            config,
        })
    }

    /// Allocate memory with ultra-high efficiency and SciRS2-Core optimizations
    pub fn allocate<T>(&self, size: usize) -> Result<UltraEfficientBuffer<T>>
    where
        T: Clone + Send + Sync + 'static,
    {
        let element_size = std::mem::size_of::<T>();
        let total_bytes = size * element_size;
        let size_class = Self::classify_size(total_bytes);
        let alignment = std::mem::align_of::<T>();

        // Update statistics
        {
            let mut stats = self.stats.lock().unwrap();
            stats.allocation_count += 1;
            stats.total_allocated += total_bytes;
            stats.current_usage += total_bytes;
            stats.peak_usage = stats.peak_usage.max(stats.current_usage);
        }

        // Check for memory pressure and apply optimizations
        if self.is_memory_pressure() && self.config.aggressive_optimization {
            self.apply_memory_pressure_optimizations()?;
        }

        match size_class {
            SizeClass::Huge => self.allocate_huge(size, total_bytes),
            SizeClass::Large => self.allocate_large(size, total_bytes),
            SizeClass::Medium | SizeClass::Small => {
                self.allocate_from_pool(size, element_size, size_class, alignment)
            }
        }
    }

    /// Deallocate memory with efficient pool return
    pub fn deallocate<T>(&self, buffer: UltraEfficientBuffer<T>) -> Result<()>
    where
        T: Clone + Send + Sync + 'static,
    {
        let total_bytes = buffer.capacity() * std::mem::size_of::<T>();

        // Update statistics
        {
            let mut stats = self.stats.lock().unwrap();
            stats.deallocation_count += 1;
            stats.total_freed += total_bytes;
            stats.current_usage = stats.current_usage.saturating_sub(total_bytes);
        }

        match &buffer.storage {
            BufferStorage::Pool { pool, .. } => {
                // Return to pool for reuse
                pool.return_buffer(buffer.into_raw_parts())?;

                let mut stats = self.stats.lock().unwrap();
                stats.cache_hits += 1;
            }
            BufferStorage::DiskBacked { .. } => {
                // Disk-backed buffers are automatically cleaned up
                let mut stats = self.stats.lock().unwrap();
                stats.disk_backed_ops += 1;
            }
            BufferStorage::ZeroCopy { .. } => {
                // Zero-copy buffers require special handling
                let mut stats = self.stats.lock().unwrap();
                stats.zero_copy_operations += 1;
            }
            BufferStorage::AdaptiveChunked { .. } => {
                // Adaptive chunked buffers are processed by chunk processor
                self.chunk_processor.process_deallocation(buffer.into_raw_parts())?;

                let mut stats = self.stats.lock().unwrap();
                stats.adaptive_chunking_ops += 1;
            }
        }

        Ok(())
    }

    /// Get memory usage statistics
    pub fn get_stats(&self) -> MemoryStats {
        self.stats.lock().unwrap().clone()
    }

    /// Perform background cleanup and optimization
    pub fn cleanup(&self) -> Result<()> {
        let start_time = Instant::now();

        // Clean up unused pools
        {
            let mut pools = self.pools.write().unwrap();
            pools.retain(|_, pool| !pool.is_empty());
        }

        // Run global pool cleanup
        self.global_pool.cleanup()?;

        // Process adaptive chunking cleanup
        self.chunk_processor.cleanup()?;

        // Update statistics
        {
            let mut stats = self.stats.lock().unwrap();
            stats.last_cleanup = Some(start_time);

            // Calculate fragmentation ratio
            if stats.total_allocated > 0 {
                stats.fragmentation_ratio =
                    (stats.total_allocated - stats.total_freed) as f64 / stats.total_allocated as f64;
            }
        }

        Ok(())
    }

    /// Check if system is under memory pressure
    fn is_memory_pressure(&self) -> bool {
        let stats = self.stats.lock().unwrap();
        let usage_ratio = stats.current_usage as f64 / self.config.max_pool_size as f64;
        usage_ratio > self.config.memory_pressure_threshold
    }

    /// Apply aggressive memory optimizations under pressure
    fn apply_memory_pressure_optimizations(&self) -> Result<()> {
        // Force cleanup
        self.cleanup()?;

        // Trigger garbage collection in pools
        {
            let pools = self.pools.read().unwrap();
            for pool in pools.values() {
                pool.force_cleanup()?;
            }
        }

        // Enable more aggressive chunking
        self.chunk_processor.enable_aggressive_mode()?;

        Ok(())
    }

    /// Classify allocation size for optimal pool selection
    fn classify_size(bytes: usize) -> SizeClass {
        match bytes {
            0..=65536 => SizeClass::Small,        // 0 - 64KB
            65537..=16777216 => SizeClass::Medium, // 64KB - 16MB
            16777217..=536870912 => SizeClass::Large, // 16MB - 512MB
            _ => SizeClass::Huge,                 // > 512MB
        }
    }

    /// Allocate from appropriate buffer pool
    fn allocate_from_pool<T>(
        &self,
        size: usize,
        element_size: usize,
        size_class: SizeClass,
        alignment: usize,
    ) -> Result<UltraEfficientBuffer<T>>
    where
        T: Clone + Send + Sync + 'static,
    {
        let pool_key = PoolKey {
            element_size,
            size_class: size_class.clone(),
            alignment,
        };

        // Get or create pool
        let pool = {
            let pools = self.pools.read().unwrap();
            if let Some(pool) = pools.get(&pool_key) {
                pool.clone()
            } else {
                drop(pools);

                // Create new pool
                let mut pools = self.pools.write().unwrap();
                let pool = Arc::new(BufferPool::new(
                    self.config.max_pool_size / 16, // Divide among multiple pools
                    element_size,
                )?);
                pools.insert(pool_key.clone(), pool.clone());
                pool
            }
        };

        // Allocate from pool
        let raw_buffer = pool.allocate(size * element_size)?;

        // Update cache stats
        {
            let mut stats = self.stats.lock().unwrap();
            stats.cache_hits += 1;
        }

        Ok(UltraEfficientBuffer {
            storage: BufferStorage::Pool {
                pool: pool.clone(),
                raw_buffer,
            },
            size,
            capacity: size,
        })
    }

    /// Allocate large buffer with adaptive chunking
    fn allocate_large<T>(&self, size: usize, total_bytes: usize) -> Result<UltraEfficientBuffer<T>>
    where
        T: Clone + Send + Sync + 'static,
    {
        if self.config.enable_adaptive_chunking {
            let chunked_array = ChunkedArray::new(size, self.chunk_processor.optimal_chunk_size())?;

            Ok(UltraEfficientBuffer {
                storage: BufferStorage::AdaptiveChunked {
                    array: chunked_array,
                },
                size,
                capacity: size,
            })
        } else {
            // Use global pool for large allocations
            let raw_buffer = self.global_pool.allocate(total_bytes)?;

            Ok(UltraEfficientBuffer {
                storage: BufferStorage::Pool {
                    pool: Arc::new(BufferPool::from_global(&self.global_pool)?),
                    raw_buffer,
                },
                size,
                capacity: size,
            })
        }
    }

    /// Allocate huge buffer with disk backing
    fn allocate_huge<T>(&self, size: usize, total_bytes: usize) -> Result<UltraEfficientBuffer<T>>
    where
        T: Clone + Send + Sync + 'static,
    {
        // Use disk-backed array for huge allocations to prevent OOM
        let disk_backed = DiskBackedArray::new(size)?;

        // Update stats
        {
            let mut stats = self.stats.lock().unwrap();
            stats.disk_backed_ops += 1;
        }

        Ok(UltraEfficientBuffer {
            storage: BufferStorage::DiskBacked {
                array: disk_backed,
            },
            size,
            capacity: size,
        })
    }

    /// Create zero-copy buffer from existing data
    pub fn create_zero_copy<T>(&self, data: &[T]) -> Result<UltraEfficientBuffer<T>>
    where
        T: Clone + Send + Sync + 'static,
    {
        if !self.config.enable_zero_copy {
            return Err(TensorError::unsupported_operation_simple(
                "Zero-copy operations disabled in pool configuration".to_string()
            ));
        }

        let zero_copy_ops = ZeroCopyOps::from_slice(data)?;

        // Update stats
        {
            let mut stats = self.stats.lock().unwrap();
            stats.zero_copy_operations += 1;
        }

        Ok(UltraEfficientBuffer {
            storage: BufferStorage::ZeroCopy {
                ops: zero_copy_ops,
            },
            size: data.len(),
            capacity: data.len(),
        })
    }

    /// Get optimal buffer size recommendation
    pub fn optimal_size_for<T>(&self, requested_size: usize) -> usize
    where
        T: 'static,
    {
        let element_size = std::mem::size_of::<T>();
        let total_bytes = requested_size * element_size;
        let size_class = Self::classify_size(total_bytes);

        match size_class {
            SizeClass::Small => {
                // Round up to next power of 2 for small allocations
                (requested_size.next_power_of_two()).max(64)
            }
            SizeClass::Medium => {
                // Round up to next 4KB boundary
                ((requested_size + 1023) / 1024) * 1024
            }
            SizeClass::Large => {
                // Round up to next MB boundary
                ((requested_size + 262143) / 262144) * 262144
            }
            SizeClass::Huge => {
                // Use adaptive chunking size
                self.chunk_processor.optimal_chunk_size()
            }
        }
    }

    /// Force memory optimization and defragmentation
    pub fn optimize(&self) -> Result<()> {
        self.apply_memory_pressure_optimizations()?;

        // Run defragmentation on all pools
        {
            let pools = self.pools.read().unwrap();
            for pool in pools.values() {
                pool.defragment()?;
            }
        }

        // Optimize global pool
        self.global_pool.optimize()?;

        Ok(())
    }
}

/// Ultra-efficient buffer with multiple storage strategies
pub struct UltraEfficientBuffer<T> {
    storage: BufferStorage<T>,
    size: usize,
    capacity: usize,
}

/// Different storage strategies for maximum efficiency
enum BufferStorage<T> {
    /// Pool-managed buffer for optimal reuse
    Pool {
        pool: Arc<BufferPool>,
        raw_buffer: *mut T,
    },

    /// Disk-backed buffer for huge allocations
    DiskBacked {
        array: DiskBackedArray<T>,
    },

    /// Zero-copy buffer for efficient data sharing
    ZeroCopy {
        ops: ZeroCopyOps<T>,
    },

    /// Adaptive chunked buffer for large tensors
    AdaptiveChunked {
        array: ChunkedArray<T>,
    },
}

impl<T> UltraEfficientBuffer<T> {
    /// Get buffer size
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get buffer capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get raw pointer for direct access (unsafe but fast)
    pub unsafe fn as_ptr(&self) -> *const T {
        match &self.storage {
            BufferStorage::Pool { raw_buffer, .. } => *raw_buffer,
            BufferStorage::DiskBacked { array } => array.as_ptr(),
            BufferStorage::ZeroCopy { ops } => ops.as_ptr(),
            BufferStorage::AdaptiveChunked { array } => array.as_ptr(),
        }
    }

    /// Get mutable raw pointer for direct access (unsafe but fast)
    pub unsafe fn as_mut_ptr(&mut self) -> *mut T {
        match &mut self.storage {
            BufferStorage::Pool { raw_buffer, .. } => *raw_buffer,
            BufferStorage::DiskBacked { array } => array.as_mut_ptr(),
            BufferStorage::ZeroCopy { ops } => ops.as_mut_ptr(),
            BufferStorage::AdaptiveChunked { array } => array.as_mut_ptr(),
        }
    }

    /// Convert to slice for safe access
    pub fn as_slice(&self) -> &[T] {
        unsafe {
            std::slice::from_raw_parts(self.as_ptr(), self.size)
        }
    }

    /// Convert to mutable slice for safe access
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe {
            std::slice::from_raw_parts_mut(self.as_mut_ptr(), self.size)
        }
    }

    /// Extract raw parts for pool return
    fn into_raw_parts(self) -> (*mut T, usize, usize) {
        let ptr = unsafe { self.as_ptr() as *mut T };
        let size = self.size;
        let capacity = self.capacity;
        std::mem::forget(self); // Prevent drop
        (ptr, size, capacity)
    }

    /// Check if buffer uses zero-copy optimization
    pub fn is_zero_copy(&self) -> bool {
        matches!(self.storage, BufferStorage::ZeroCopy { .. })
    }

    /// Check if buffer is disk-backed
    pub fn is_disk_backed(&self) -> bool {
        matches!(self.storage, BufferStorage::DiskBacked { .. })
    }

    /// Check if buffer uses adaptive chunking
    pub fn is_adaptive_chunked(&self) -> bool {
        matches!(self.storage, BufferStorage::AdaptiveChunked { .. })
    }
}

/// Global memory pool instance for efficient sharing
static GLOBAL_MEMORY_POOL: std::sync::OnceLock<UltraEfficientMemoryPool> = std::sync::OnceLock::new();

/// Get or initialize the global memory pool
pub fn global_memory_pool() -> &'static UltraEfficientMemoryPool {
    GLOBAL_MEMORY_POOL.get_or_init(|| {
        UltraEfficientMemoryPool::new(PoolConfig::default())
            .expect("Failed to initialize global memory pool")
    })
}

/// Performance monitoring and profiling utilities
pub mod profiling {
    use super::*;
    use std::time::Instant;

    /// Memory allocation profiler
    pub struct MemoryProfiler {
        start_time: Instant,
        initial_stats: MemoryStats,
    }

    impl MemoryProfiler {
        /// Start profiling memory operations
        pub fn start() -> Self {
            let pool = global_memory_pool();
            Self {
                start_time: Instant::now(),
                initial_stats: pool.get_stats(),
            }
        }

        /// Finish profiling and get report
        pub fn finish(self) -> MemoryProfileReport {
            let pool = global_memory_pool();
            let final_stats = pool.get_stats();
            let duration = self.start_time.elapsed();

            MemoryProfileReport {
                duration,
                initial_stats: self.initial_stats,
                final_stats,
            }
        }
    }

    /// Memory profiling report
    pub struct MemoryProfileReport {
        pub duration: Duration,
        pub initial_stats: MemoryStats,
        pub final_stats: MemoryStats,
    }

    impl MemoryProfileReport {
        /// Print detailed memory usage report
        pub fn print_report(&self) {
            println!("🚀 === ULTRA-EFFICIENT MEMORY POOL REPORT ===");
            println!("Duration: {:.3}ms", self.duration.as_secs_f64() * 1000.0);

            let alloc_delta = self.final_stats.allocation_count - self.initial_stats.allocation_count;
            let dealloc_delta = self.final_stats.deallocation_count - self.initial_stats.deallocation_count;
            let memory_delta = self.final_stats.current_usage as i64 - self.initial_stats.current_usage as i64;

            println!("Allocations: {} (+{})", self.final_stats.allocation_count, alloc_delta);
            println!("Deallocations: {} (+{})", self.final_stats.deallocation_count, dealloc_delta);
            println!("Memory Delta: {:+} bytes", memory_delta);
            println!("Peak Usage: {} bytes", self.final_stats.peak_usage);
            println!("Cache Hit Rate: {:.2}%",
                     100.0 * self.final_stats.cache_hits as f64 /
                     (self.final_stats.cache_hits + self.final_stats.cache_misses) as f64);
            println!("Fragmentation: {:.2}%", self.final_stats.fragmentation_ratio * 100.0);
            println!("Zero-Copy Operations: {}", self.final_stats.zero_copy_operations);
            println!("Adaptive Chunking Operations: {}", self.final_stats.adaptive_chunking_ops);
            println!("Disk-Backed Operations: {}", self.final_stats.disk_backed_ops);
        }

        /// Get performance metrics
        pub fn performance_metrics(&self) -> MemoryPerformanceMetrics {
            let total_operations = self.final_stats.allocation_count + self.final_stats.deallocation_count;
            let ops_per_second = if self.duration.as_secs_f64() > 0.0 {
                total_operations as f64 / self.duration.as_secs_f64()
            } else {
                0.0
            };

            let cache_hit_rate = if self.final_stats.cache_hits + self.final_stats.cache_misses > 0 {
                self.final_stats.cache_hits as f64 /
                (self.final_stats.cache_hits + self.final_stats.cache_misses) as f64
            } else {
                0.0
            };

            MemoryPerformanceMetrics {
                operations_per_second: ops_per_second,
                cache_hit_rate,
                fragmentation_ratio: self.final_stats.fragmentation_ratio,
                zero_copy_ratio: self.final_stats.zero_copy_operations as f64 / total_operations.max(1) as f64,
                memory_efficiency: 1.0 - self.final_stats.fragmentation_ratio,
            }
        }
    }

    /// Performance metrics for memory operations
    #[derive(Debug, Clone)]
    pub struct MemoryPerformanceMetrics {
        pub operations_per_second: f64,
        pub cache_hit_rate: f64,
        pub fragmentation_ratio: f64,
        pub zero_copy_ratio: f64,
        pub memory_efficiency: f64,
    }
}

// Make buffer safe to send between threads
unsafe impl<T: Send> Send for UltraEfficientBuffer<T> {}
unsafe impl<T: Sync> Sync for UltraEfficientBuffer<T> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool_basic_allocation() {
        let pool = UltraEfficientMemoryPool::new(PoolConfig::default()).unwrap();
        let buffer: UltraEfficientBuffer<f32> = pool.allocate(1000).unwrap();

        assert_eq!(buffer.size(), 1000);
        assert!(buffer.capacity() >= 1000);

        pool.deallocate(buffer).unwrap();
    }

    #[test]
    fn test_size_classification() {
        assert_eq!(UltraEfficientMemoryPool::classify_size(1000), SizeClass::Small);
        assert_eq!(UltraEfficientMemoryPool::classify_size(100_000), SizeClass::Medium);
        assert_eq!(UltraEfficientMemoryPool::classify_size(20_000_000), SizeClass::Large);
        assert_eq!(UltraEfficientMemoryPool::classify_size(1_000_000_000), SizeClass::Huge);
    }

    #[test]
    fn test_memory_stats() {
        let pool = UltraEfficientMemoryPool::new(PoolConfig::default()).unwrap();
        let initial_stats = pool.get_stats();

        let buffer: UltraEfficientBuffer<i32> = pool.allocate(500).unwrap();
        let stats_after_alloc = pool.get_stats();

        assert!(stats_after_alloc.allocation_count > initial_stats.allocation_count);
        assert!(stats_after_alloc.current_usage > initial_stats.current_usage);

        pool.deallocate(buffer).unwrap();
        let final_stats = pool.get_stats();

        assert!(final_stats.deallocation_count > initial_stats.deallocation_count);
    }

    #[test]
    fn test_optimal_size_calculation() {
        let pool = UltraEfficientMemoryPool::new(PoolConfig::default()).unwrap();

        let optimal_small = pool.optimal_size_for::<f32>(100);
        let optimal_medium = pool.optimal_size_for::<f32>(50_000);

        assert!(optimal_small >= 100);
        assert!(optimal_medium >= 50_000);
        assert!(optimal_small.is_power_of_two() || optimal_small >= 64);
    }

    #[test]
    fn test_memory_pressure_handling() {
        let mut config = PoolConfig::default();
        config.memory_pressure_threshold = 0.1; // Very low threshold for testing
        config.max_pool_size = 1024; // Small pool for testing

        let pool = UltraEfficientMemoryPool::new(config).unwrap();

        // Allocate enough to trigger pressure
        let _buffers: Vec<UltraEfficientBuffer<u8>> = (0..100)
            .map(|_| pool.allocate(100).unwrap())
            .collect();

        // Should handle pressure gracefully
        let cleanup_result = pool.cleanup();
        assert!(cleanup_result.is_ok());
    }

    #[test]
    fn test_profiling() {
        let profiler = profiling::MemoryProfiler::start();

        let pool = global_memory_pool();
        let buffer: UltraEfficientBuffer<f64> = pool.allocate(1000).unwrap();
        pool.deallocate(buffer).unwrap();

        let report = profiler.finish();
        let metrics = report.performance_metrics();

        assert!(metrics.operations_per_second >= 0.0);
        assert!(metrics.cache_hit_rate >= 0.0 && metrics.cache_hit_rate <= 1.0);
        assert!(metrics.memory_efficiency >= 0.0 && metrics.memory_efficiency <= 1.0);
    }
}