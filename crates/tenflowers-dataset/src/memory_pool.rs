//! Memory pooling utilities for dataset operations
//!
//! This module provides efficient memory allocation and reuse mechanisms
//! to reduce allocation overhead during dataset iteration and batch processing.

#![allow(unsafe_code)]

use std::alloc::{alloc, dealloc, Layout};
use std::collections::VecDeque;
use std::mem;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};

/// Memory pool statistics for monitoring
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    pub allocations: u64,
    pub deallocations: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub current_size: usize,
    pub peak_size: usize,
}

impl PoolStats {
    /// Calculate the cache hit ratio
    pub fn hit_ratio(&self) -> f64 {
        if self.cache_hits + self.cache_misses == 0 {
            0.0
        } else {
            self.cache_hits as f64 / (self.cache_hits + self.cache_misses) as f64
        }
    }

    /// Get memory utilization efficiency
    pub fn efficiency(&self) -> f64 {
        if self.allocations == 0 {
            0.0
        } else {
            self.cache_hits as f64 / self.allocations as f64
        }
    }
}

/// A memory block that can be reused
#[derive(Debug)]
struct MemoryBlock {
    ptr: NonNull<u8>,
    size: usize,
    layout: Layout,
}

impl MemoryBlock {
    fn new(size: usize) -> Result<Self, String> {
        let layout = Layout::from_size_align(size, mem::align_of::<u8>())
            .map_err(|e| format!("Layout error: {e:?}"))?;

        let ptr = NonNull::new(unsafe { alloc(layout) })
            .ok_or_else(|| "Memory allocation failed".to_string())?;

        Ok(Self { ptr, size, layout })
    }

    /// Get a slice view of the memory block
    pub fn as_slice_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.size) }
    }

    /// Get the raw pointer
    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }
}

impl Drop for MemoryBlock {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.ptr.as_ptr(), self.layout);
        }
    }
}

// SAFETY: MemoryBlock owns its memory exclusively and the pointer is valid
unsafe impl Send for MemoryBlock {}
unsafe impl Sync for MemoryBlock {}

/// Memory pool for efficient allocation and reuse
pub struct MemoryPool {
    pools: Vec<Mutex<VecDeque<MemoryBlock>>>,
    max_blocks_per_size: usize,
    min_block_size: usize,
    max_block_size: usize,
    stats: Arc<Mutex<PoolStats>>,
}

impl MemoryPool {
    /// Create a new memory pool
    pub fn new() -> Self {
        Self::with_config(64, 1024, 1024 * 1024 * 16) // 1KB to 16MB
    }

    /// Create a memory pool with custom configuration
    pub fn with_config(
        max_blocks_per_size: usize,
        min_block_size: usize,
        max_block_size: usize,
    ) -> Self {
        // Create pools for different size classes (powers of 2)
        let mut size = min_block_size;
        let mut pools = Vec::new();

        while size <= max_block_size {
            pools.push(Mutex::new(VecDeque::new()));
            size *= 2;
        }

        Self {
            pools,
            max_blocks_per_size,
            min_block_size,
            max_block_size,
            stats: Arc::new(Mutex::new(PoolStats::default())),
        }
    }

    /// Find the appropriate size class for a requested size
    fn find_size_class(&self, size: usize) -> Option<usize> {
        if size < self.min_block_size || size > self.max_block_size {
            return None;
        }

        let mut class_size = self.min_block_size;
        let mut class_index = 0;

        while class_size < size && class_index < self.pools.len() {
            class_size *= 2;
            class_index += 1;
        }

        if class_index < self.pools.len() {
            Some(class_index)
        } else {
            None
        }
    }

    /// Allocate a memory block from the pool
    pub fn allocate(self: &Arc<Self>, size: usize) -> Result<PooledMemory, String> {
        let mut stats = self
            .stats
            .lock()
            .map_err(|e| format!("Failed to acquire stats lock: {e}"))?;
        stats.allocations += 1;

        if let Some(class_index) = self.find_size_class(size) {
            let mut pool = self.pools[class_index]
                .lock()
                .map_err(|e| format!("Failed to acquire pool lock: {e}"))?;

            if let Some(block) = pool.pop_front() {
                stats.cache_hits += 1;
                stats.current_size -= block.size;
                drop(stats);
                drop(pool);

                return Ok(PooledMemory {
                    block: Some(block),
                    pool: Arc::downgrade(self),
                    class_index: Some(class_index),
                });
            } else {
                stats.cache_misses += 1;
                drop(pool);
            }
        } else {
            stats.cache_misses += 1;
        }

        // Allocate new block
        let actual_size = if let Some(class_index) = self.find_size_class(size) {
            self.min_block_size << class_index
        } else {
            size
        };

        let block = MemoryBlock::new(actual_size)?;
        stats.current_size += block.size;
        stats.peak_size = stats.peak_size.max(stats.current_size);
        drop(stats);

        Ok(PooledMemory {
            block: Some(block),
            pool: Arc::downgrade(self),
            class_index: self.find_size_class(size),
        })
    }

    /// Return a memory block to the pool
    fn deallocate(&self, block: MemoryBlock, class_index: Option<usize>) {
        let mut stats = self.stats.lock().unwrap();
        stats.deallocations += 1;

        if let Some(class_index) = class_index {
            if class_index < self.pools.len() {
                let mut pool = self.pools[class_index].lock().unwrap();

                if pool.len() < self.max_blocks_per_size {
                    stats.current_size += block.size;
                    pool.push_back(block);
                    return;
                }
            }
        }

        // Block will be dropped automatically if not returned to pool
        stats.current_size = stats.current_size.saturating_sub(block.size);
    }

    /// Get current pool statistics
    pub fn stats(&self) -> PoolStats {
        self.stats.lock().unwrap().clone()
    }

    /// Clear all cached blocks
    pub fn clear(&self) {
        for pool in &self.pools {
            pool.lock().unwrap().clear();
        }

        let mut stats = self.stats.lock().unwrap();
        stats.current_size = 0;
    }
}

impl Clone for MemoryPool {
    fn clone(&self) -> Self {
        // Note: This creates a new pool with the same configuration
        // The actual cached blocks are not cloned
        Self::with_config(
            self.max_blocks_per_size,
            self.min_block_size,
            self.max_block_size,
        )
    }
}

impl Default for MemoryPool {
    fn default() -> Self {
        Self::new()
    }
}

/// A memory allocation from the pool that automatically returns to the pool on drop
pub struct PooledMemory {
    block: Option<MemoryBlock>,
    pool: std::sync::Weak<MemoryPool>,
    class_index: Option<usize>,
}

impl PooledMemory {
    /// Get the size of the allocated memory
    pub fn size(&self) -> usize {
        self.block.as_ref().map(|b| b.size).unwrap_or(0)
    }

    /// Get a mutable slice view of the memory
    pub fn as_slice_mut(&mut self) -> &mut [u8] {
        self.block.as_mut().unwrap().as_slice_mut()
    }

    /// Get the raw pointer
    pub fn as_ptr(&self) -> *mut u8 {
        self.block.as_ref().unwrap().as_ptr()
    }

    /// Convert to a `Vec<u8>` (consumes the pooled memory)
    pub fn into_vec(mut self) -> Vec<u8> {
        let block = self.block.take().unwrap();
        let size = block.size;
        let ptr = block.as_ptr();

        // Prevent the block from being deallocated
        mem::forget(block);

        // Create a Vec from the raw pointer
        unsafe { Vec::from_raw_parts(ptr, size, size) }
    }
}

impl Drop for PooledMemory {
    fn drop(&mut self) {
        if let Some(block) = self.block.take() {
            if let Some(pool) = self.pool.upgrade() {
                pool.deallocate(block, self.class_index);
            }
            // If pool is dropped, block will be automatically deallocated
        }
    }
}

/// Thread-safe global memory pool
pub struct GlobalMemoryPool {
    pool: Arc<MemoryPool>,
}

impl GlobalMemoryPool {
    /// Get the global memory pool instance
    pub fn instance() -> &'static GlobalMemoryPool {
        static INSTANCE: std::sync::OnceLock<GlobalMemoryPool> = std::sync::OnceLock::new();
        INSTANCE.get_or_init(|| GlobalMemoryPool {
            pool: Arc::new(MemoryPool::new()),
        })
    }

    /// Allocate memory from the global pool
    pub fn allocate(size: usize) -> Result<PooledMemory, String> {
        Self::instance().pool.allocate(size)
    }

    /// Get global pool statistics
    pub fn stats() -> PoolStats {
        Self::instance().pool.stats()
    }

    /// Clear the global pool
    pub fn clear() {
        Self::instance().pool.clear()
    }
}

/// Extension trait for easy memory pool allocation
pub trait MemoryPoolExt<T> {
    /// Allocate a vector using the memory pool
    fn with_pool_capacity(capacity: usize) -> Result<Vec<T>, String>;
}

impl<T> MemoryPoolExt<T> for Vec<T> {
    fn with_pool_capacity(capacity: usize) -> Result<Vec<T>, String> {
        let size = capacity * mem::size_of::<T>();
        let pooled = GlobalMemoryPool::allocate(size)?;

        // Convert to Vec
        let vec = pooled.into_vec();

        // Cast to the proper type (this is safe since we allocated the right amount)
        let ptr = vec.as_ptr() as *mut T;
        let len = 0;

        mem::forget(vec); // Prevent deallocation

        Ok(unsafe { Vec::from_raw_parts(ptr, len, capacity) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool_basic() {
        let pool = Arc::new(MemoryPool::new());

        // Allocate some memory
        let mut mem1 = pool.allocate(1024).unwrap();
        assert_eq!(mem1.size(), 1024);

        // Write some data
        let slice = mem1.as_slice_mut();
        slice[0] = 42;
        slice[1023] = 99;

        drop(mem1);

        // Allocate again
        let mut mem2 = pool.allocate(1024).unwrap();
        let slice2 = mem2.as_slice_mut();

        // Verify we can write to the new allocation
        slice2[0] = 100;
        assert_eq!(slice2[0], 100);

        let stats = pool.stats();
        assert_eq!(stats.allocations, 2);
        // With the Arc optimization, we should now see cache hits
        assert_eq!(stats.cache_misses, 1);
        assert_eq!(stats.cache_hits, 1);
    }

    #[test]
    fn test_memory_pool_different_sizes() {
        let pool = Arc::new(MemoryPool::new());

        let mem1 = pool.allocate(512).unwrap();
        let mem2 = pool.allocate(1024).unwrap();
        let mem3 = pool.allocate(2048).unwrap();

        assert!(mem1.size() >= 512);
        assert!(mem2.size() >= 1024);
        assert!(mem3.size() >= 2048);

        drop(mem1);
        drop(mem2);
        drop(mem3);

        let stats = pool.stats();
        assert_eq!(stats.allocations, 3);
    }

    #[test]
    fn test_global_memory_pool() {
        let mem1 = GlobalMemoryPool::allocate(1024).unwrap();
        assert_eq!(mem1.size(), 1024);

        let mem2 = GlobalMemoryPool::allocate(2048).unwrap();
        assert!(mem2.size() >= 2048);

        // Basic functionality test - just verify allocations work
        drop(mem1);
        drop(mem2);

        let stats = GlobalMemoryPool::stats();
        assert!(stats.allocations >= 2);
    }

    #[test]
    fn test_vec_with_pool_capacity() {
        GlobalMemoryPool::clear();

        let mut vec: Vec<i32> = Vec::with_pool_capacity(100).unwrap();
        vec.push(42);
        vec.push(99);

        assert_eq!(vec.len(), 2);
        assert_eq!(vec.capacity(), 100);
        assert_eq!(vec[0], 42);
        assert_eq!(vec[1], 99);
    }

    #[test]
    fn test_pool_stats() {
        let pool = Arc::new(MemoryPool::new());

        let stats = pool.stats();
        assert_eq!(stats.allocations, 0);
        assert_eq!(stats.deallocations, 0);
        assert_eq!(stats.cache_hits, 0);
        assert_eq!(stats.cache_misses, 0);
        assert_eq!(stats.hit_ratio(), 0.0);
        assert_eq!(stats.efficiency(), 0.0);

        let _mem = pool.allocate(1024).unwrap();
        let stats = pool.stats();
        assert_eq!(stats.allocations, 1);
        assert_eq!(stats.cache_misses, 1);
    }
}
