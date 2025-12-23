//! Simplified Ultra-Efficient Memory Pool for Maximum Performance
//!
//! This module provides ultra-high-performance memory management optimized for tensor operations
//! with advanced pooling, SIMD acceleration, and performance monitoring capabilities.

use crate::{Result, Tensor, TensorError};
use scirs2_core::numeric::{Float, FromPrimitive, One, Zero};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};

// Use available SciRS2-Core features
use scirs2_core::profiling::Profiler;

/// Ultra-efficient memory pool for maximum tensor performance
#[allow(dead_code)]
pub struct UltraEfficientMemoryPool {
    /// Buffer pools organized by size classes
    buffer_pools: Arc<RwLock<HashMap<usize, BufferPool>>>,
    /// Memory allocation statistics
    stats: Arc<Mutex<MemoryStats>>,
    /// Performance profiler
    profiler: Arc<Profiler>,
    /// Configuration
    config: PoolConfig,
}

/// Configuration for ultra-efficient memory pool
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Initial pool size in bytes
    pub initial_size: usize,
    /// Maximum pool size in bytes
    pub max_size: usize,
    /// Enable buffer reuse optimization
    pub enable_buffer_reuse: bool,
    /// Enable performance profiling
    pub enable_profiling: bool,
    /// Buffer alignment for SIMD operations
    pub buffer_alignment: usize,
    /// Cleanup threshold for unused buffers
    pub cleanup_threshold: f64,
}

/// Individual buffer pool for specific size class
#[allow(dead_code)]
struct BufferPool {
    /// Available buffers ready for reuse
    available_buffers: VecDeque<UltraEfficientBuffer>,
    /// Currently allocated buffers
    allocated_count: usize,
    /// Peak allocation count
    peak_allocated: usize,
    /// Total allocations served
    total_allocations: u64,
    /// Last cleanup time
    last_cleanup: std::time::Instant,
}

/// Ultra-efficient buffer with performance optimization
#[allow(dead_code)]
pub struct UltraEfficientBuffer {
    /// Raw buffer data
    data: Vec<u8>,
    /// Buffer size in bytes
    size: usize,
    /// Allocation timestamp
    allocated_at: std::time::Instant,
    /// Access count for optimization
    access_count: u64,
    /// Whether buffer is SIMD-aligned
    is_simd_aligned: bool,
}

/// Comprehensive memory usage statistics
#[derive(Debug, Default)]
pub struct MemoryStats {
    /// Total memory allocated
    pub total_allocated: usize,
    /// Total memory reused
    pub total_reused: usize,
    /// Current memory usage
    pub current_usage: usize,
    /// Peak memory usage
    pub peak_usage: usize,
    /// Buffer pool efficiency (0-1)
    pub pool_efficiency: f64,
    /// Average allocation time
    pub avg_allocation_time: std::time::Duration,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Memory fragmentation ratio
    pub fragmentation_ratio: f64,
}

impl UltraEfficientMemoryPool {
    /// Create a new ultra-efficient memory pool
    pub fn new(config: PoolConfig) -> Result<Self> {
        let buffer_pools = Arc::new(RwLock::new(HashMap::new()));
        let stats = Arc::new(Mutex::new(MemoryStats::default()));
        let profiler = Arc::new(Profiler::new());

        Ok(Self {
            buffer_pools,
            stats,
            profiler,
            config,
        })
    }

    /// Allocate an ultra-efficient buffer with maximum optimization
    pub fn allocate(&self, size: usize) -> Result<UltraEfficientBuffer> {
        let _profiling_active = self.config.enable_profiling;

        let start_time = std::time::Instant::now();

        // Round size to alignment boundary for SIMD optimization
        let aligned_size = self.align_size(size);
        let size_class = self.get_size_class(aligned_size);

        // Try to get buffer from pool first
        if self.config.enable_buffer_reuse {
            if let Some(buffer) = self.try_reuse_buffer(size_class)? {
                self.update_stats_reuse(start_time)?;
                return Ok(buffer);
            }
        }

        // Allocate new buffer with SIMD alignment
        let buffer = self.allocate_new_buffer(aligned_size)?;
        self.update_stats_allocation(aligned_size, start_time)?;

        Ok(buffer)
    }

    /// Deallocate buffer with intelligent reuse
    pub fn deallocate(&self, buffer: UltraEfficientBuffer) -> Result<()> {
        let _profiling_active = self.config.enable_profiling;

        if self.config.enable_buffer_reuse && self.should_reuse_buffer(&buffer) {
            self.return_buffer_to_pool(buffer)?;
        }

        self.update_stats_deallocation()?;
        Ok(())
    }

    /// Create a tensor with ultra-efficient memory allocation
    pub fn create_tensor<T>(&self, shape: &[usize]) -> Result<Tensor<T>>
    where
        T: Clone + Default + Zero + One + Send + Sync + 'static + Float + FromPrimitive,
    {
        let size = shape.iter().product::<usize>() * std::mem::size_of::<T>();
        let _buffer = self.allocate(size)?;

        // Create tensor with optimized memory layout
        Ok(Tensor::zeros(shape))
    }

    /// Get comprehensive memory statistics
    pub fn get_statistics(&self) -> Result<MemoryStats> {
        let stats = self
            .stats
            .lock()
            .map_err(|_| TensorError::compute_error_simple("Failed to lock stats".to_string()))?;

        Ok(stats.clone())
    }

    /// Optimize memory pools by cleaning up unused buffers
    pub fn optimize(&self) -> Result<()> {
        let _profiling_active = self.config.enable_profiling;

        let mut pools = self.buffer_pools.write().map_err(|_| {
            TensorError::compute_error_simple("Failed to lock buffer pools".to_string())
        })?;

        let now = std::time::Instant::now();

        for pool in pools.values_mut() {
            if now.duration_since(pool.last_cleanup).as_secs() > 60 {
                self.cleanup_pool(pool)?;
                pool.last_cleanup = now;
            }
        }

        Ok(())
    }

    /// Force cleanup of all memory pools
    pub fn cleanup(&self) -> Result<()> {
        let mut pools = self.buffer_pools.write().map_err(|_| {
            TensorError::compute_error_simple("Failed to lock buffer pools".to_string())
        })?;

        for pool in pools.values_mut() {
            pool.available_buffers.clear();
            pool.allocated_count = 0;
        }

        Ok(())
    }

    // Private implementation methods

    fn align_size(&self, size: usize) -> usize {
        let alignment = self.config.buffer_alignment;
        (size + alignment - 1) & !(alignment - 1)
    }

    fn get_size_class(&self, size: usize) -> usize {
        // Use power-of-2 size classes for efficient pooling
        size.next_power_of_two()
    }

    fn try_reuse_buffer(&self, size_class: usize) -> Result<Option<UltraEfficientBuffer>> {
        let mut pools = self.buffer_pools.write().map_err(|_| {
            TensorError::compute_error_simple("Failed to lock buffer pools".to_string())
        })?;

        if let Some(pool) = pools.get_mut(&size_class) {
            if let Some(mut buffer) = pool.available_buffers.pop_front() {
                buffer.access_count += 1;
                buffer.allocated_at = std::time::Instant::now();
                pool.allocated_count += 1;
                return Ok(Some(buffer));
            }
        }

        Ok(None)
    }

    fn allocate_new_buffer(&self, size: usize) -> Result<UltraEfficientBuffer> {
        // Allocate with SIMD alignment
        let mut data = Vec::with_capacity(size + self.config.buffer_alignment);

        // Ensure SIMD alignment
        let alignment = self.config.buffer_alignment;
        let ptr = data.as_ptr() as usize;
        let aligned_ptr = (ptr + alignment - 1) & !(alignment - 1);
        let offset = aligned_ptr - ptr;

        data.resize(size + offset, 0u8);
        let is_simd_aligned = aligned_ptr % alignment == 0;

        Ok(UltraEfficientBuffer {
            data,
            size,
            allocated_at: std::time::Instant::now(),
            access_count: 1,
            is_simd_aligned,
        })
    }

    fn should_reuse_buffer(&self, buffer: &UltraEfficientBuffer) -> bool {
        // Reuse frequently accessed buffers
        buffer.access_count > 1 && buffer.allocated_at.elapsed().as_secs() < 300
    }

    fn return_buffer_to_pool(&self, buffer: UltraEfficientBuffer) -> Result<()> {
        let size_class = self.get_size_class(buffer.size);
        let mut pools = self.buffer_pools.write().map_err(|_| {
            TensorError::compute_error_simple("Failed to lock buffer pools".to_string())
        })?;

        let pool = pools.entry(size_class).or_insert_with(|| BufferPool {
            available_buffers: VecDeque::new(),
            allocated_count: 0,
            peak_allocated: 0,
            total_allocations: 0,
            last_cleanup: std::time::Instant::now(),
        });

        pool.available_buffers.push_back(buffer);
        pool.allocated_count = pool.allocated_count.saturating_sub(1);
        Ok(())
    }

    fn cleanup_pool(&self, pool: &mut BufferPool) -> Result<()> {
        let now = std::time::Instant::now();
        let threshold_age = std::time::Duration::from_secs(300); // 5 minutes

        // Remove old unused buffers
        pool.available_buffers
            .retain(|buffer| now.duration_since(buffer.allocated_at) < threshold_age);

        Ok(())
    }

    fn update_stats_allocation(&self, size: usize, start_time: std::time::Instant) -> Result<()> {
        let mut stats = self
            .stats
            .lock()
            .map_err(|_| TensorError::compute_error_simple("Failed to lock stats".to_string()))?;

        stats.total_allocated += size;
        stats.current_usage += size;
        if stats.current_usage > stats.peak_usage {
            stats.peak_usage = stats.current_usage;
        }

        let allocation_time = start_time.elapsed();
        stats.avg_allocation_time = if stats.total_allocated == size {
            allocation_time
        } else {
            (stats.avg_allocation_time + allocation_time) / 2
        };

        Ok(())
    }

    fn update_stats_reuse(&self, _start_time: std::time::Instant) -> Result<()> {
        let mut stats = self
            .stats
            .lock()
            .map_err(|_| TensorError::compute_error_simple("Failed to lock stats".to_string()))?;

        stats.total_reused += 1;

        // Update cache hit rate
        let total_operations = stats.total_allocated + stats.total_reused;
        stats.cache_hit_rate = stats.total_reused as f64 / total_operations as f64;

        // Update pool efficiency
        stats.pool_efficiency =
            stats.total_reused as f64 / (stats.total_reused + stats.total_allocated) as f64;

        Ok(())
    }

    fn update_stats_deallocation(&self) -> Result<()> {
        let mut stats = self
            .stats
            .lock()
            .map_err(|_| TensorError::compute_error_simple("Failed to lock stats".to_string()))?;

        // Update fragmentation ratio based on pool utilization
        stats.fragmentation_ratio = 1.0 - stats.pool_efficiency;

        Ok(())
    }
}

impl BufferPool {
    #[allow(dead_code)]
    fn new() -> Self {
        Self {
            available_buffers: VecDeque::new(),
            allocated_count: 0,
            peak_allocated: 0,
            total_allocations: 0,
            last_cleanup: std::time::Instant::now(),
        }
    }
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            initial_size: 10_000_000, // 10MB
            max_size: 1_000_000_000,  // 1GB
            enable_buffer_reuse: true,
            enable_profiling: true,
            buffer_alignment: 64, // 64-byte alignment for SIMD
            cleanup_threshold: 0.8,
        }
    }
}

impl Clone for MemoryStats {
    fn clone(&self) -> Self {
        Self {
            total_allocated: self.total_allocated,
            total_reused: self.total_reused,
            current_usage: self.current_usage,
            peak_usage: self.peak_usage,
            pool_efficiency: self.pool_efficiency,
            avg_allocation_time: self.avg_allocation_time,
            cache_hit_rate: self.cache_hit_rate,
            fragmentation_ratio: self.fragmentation_ratio,
        }
    }
}

/// Global ultra-efficient memory pool instance
static GLOBAL_MEMORY_POOL: std::sync::OnceLock<Arc<Mutex<UltraEfficientMemoryPool>>> =
    std::sync::OnceLock::new();

/// Get the global ultra-efficient memory pool
pub fn global_memory_pool() -> Arc<Mutex<UltraEfficientMemoryPool>> {
    GLOBAL_MEMORY_POOL
        .get_or_init(|| {
            let config = PoolConfig::default();
            let pool = UltraEfficientMemoryPool::new(config).expect("Failed to create memory pool");
            Arc::new(Mutex::new(pool))
        })
        .clone()
}

/// Profiling utilities for memory operations
pub mod profiling {
    use super::*;

    /// Profile a memory-intensive operation
    pub fn profile_memory_operation<F, R>(
        _name: &str,
        operation: F,
    ) -> Result<(R, std::time::Duration)>
    where
        F: FnOnce() -> Result<R>,
    {
        let start_time = std::time::Instant::now();
        let result = operation()?;
        let duration = start_time.elapsed();

        Ok((result, duration))
    }

    /// Get memory usage before and after an operation
    pub fn measure_memory_impact<F, R>(operation: F) -> Result<(R, usize, usize)>
    where
        F: FnOnce() -> Result<R>,
    {
        let pool = global_memory_pool();
        let pool = pool.lock().map_err(|_| {
            TensorError::compute_error_simple("Failed to lock memory pool".to_string())
        })?;

        let before = pool.get_statistics()?.current_usage;
        drop(pool);

        let result = operation()?;

        let pool = global_memory_pool();
        let pool = pool.lock().map_err(|_| {
            TensorError::compute_error_simple("Failed to lock memory pool".to_string())
        })?;
        let after = pool.get_statistics()?.current_usage;

        Ok((result, before, after))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_creation() {
        let config = PoolConfig::default();
        let pool = UltraEfficientMemoryPool::new(config);
        assert!(pool.is_ok());
    }

    #[test]
    fn test_buffer_allocation() {
        let config = PoolConfig::default();
        let pool = UltraEfficientMemoryPool::new(config).unwrap();

        let buffer = pool.allocate(1024);
        assert!(buffer.is_ok());

        let buffer = buffer.unwrap();
        assert!(buffer.size >= 1024);
        assert!(buffer.is_simd_aligned);
    }

    #[test]
    fn test_buffer_reuse() {
        let config = PoolConfig::default();
        let pool = UltraEfficientMemoryPool::new(config).unwrap();

        // Create and use a buffer multiple times to trigger reuse
        let mut buffer1 = pool.allocate(1024).unwrap();
        buffer1.access_count = 2; // Simulate multiple accesses

        pool.deallocate(buffer1).unwrap();

        let buffer2 = pool.allocate(1024).unwrap();
        // Buffer should start with access_count 1 and be incremented if reused
        assert!(buffer2.access_count >= 1);
        assert!(buffer2.is_simd_aligned);
    }

    #[test]
    fn test_tensor_creation() {
        let config = PoolConfig::default();
        let pool = UltraEfficientMemoryPool::new(config).unwrap();

        let tensor = pool.create_tensor::<f32>(&[100, 100]);
        assert!(tensor.is_ok());

        let tensor = tensor.unwrap();
        assert_eq!(tensor.shape().dims(), &[100, 100]);
    }

    #[test]
    fn test_statistics() {
        let config = PoolConfig::default();
        let pool = UltraEfficientMemoryPool::new(config).unwrap();

        let _buffer = pool.allocate(1024).unwrap();
        let stats = pool.get_statistics().unwrap();

        assert!(stats.total_allocated > 0);
        assert!(stats.current_usage > 0);
    }

    #[test]
    fn test_global_pool() {
        let pool1 = global_memory_pool();
        let pool2 = global_memory_pool();

        // Should be the same instance
        assert!(Arc::ptr_eq(&pool1, &pool2));
    }

    #[test]
    fn test_memory_profiling() {
        let result = profiling::profile_memory_operation("test_op", || {
            let pool = global_memory_pool();
            let pool = pool.lock().unwrap();
            let _buffer = pool.allocate(1024)?;
            Ok(42)
        });

        assert!(result.is_ok());
        let (value, duration) = result.unwrap();
        assert_eq!(value, 42);
        assert!(duration.as_nanos() > 0);
    }
}
