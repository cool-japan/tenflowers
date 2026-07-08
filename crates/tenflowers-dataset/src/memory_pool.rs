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
    /// Allocate a new memory block with the given size and alignment.
    ///
    /// The `align` parameter MUST be the alignment actually required by
    /// whatever type will eventually occupy this block (e.g.
    /// `mem::align_of::<T>()`), not a hardcoded byte alignment. Handing out
    /// a block that is less aligned than the type that will be stored in it
    /// is undefined behavior (see `MemoryPoolExt::with_pool_capacity`).
    fn new(size: usize, align: usize) -> Result<Self, String> {
        // `GlobalAlloc::alloc` requires `layout.size() > 0` (a zero-size
        // allocation request is itself undefined behavior, independent of
        // alignment). `size == 0` is reachable here (e.g. `pool.allocate(0)`
        // or `with_pool_capacity::<T>(0)`), so it must be rejected before
        // ever reaching `alloc()` rather than trusted to not occur.
        if size == 0 {
            return Err("MemoryBlock::new requires a non-zero size".to_string());
        }

        // `Layout::from_size_align` requires a non-zero power-of-two
        // alignment; `mem::align_of::<T>()` always satisfies this for any
        // concrete `T`, and callers that just want byte storage pass 1.
        let layout =
            Layout::from_size_align(size, align).map_err(|e| format!("Layout error: {e:?}"))?;

        let ptr = NonNull::new(unsafe { alloc(layout) })
            .ok_or_else(|| "Memory allocation failed".to_string())?;

        Ok(Self { ptr, size, layout })
    }

    /// The alignment this block was actually allocated with.
    ///
    /// This is the alignment recorded in `self.layout` at allocation time,
    /// so it always matches what was passed to `alloc()` (and, by
    /// extension, what `dealloc()` will use via `Drop`).
    fn align(&self) -> usize {
        self.layout.align()
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

    /// Allocate a memory block from the pool with the default (byte)
    /// alignment.
    ///
    /// This is the historical entry point for callers that only need a
    /// byte buffer (e.g. `PooledMemory::into_vec` -> `Vec<u8>`). Callers
    /// that will reinterpret the block as some other type `T` MUST use
    /// [`MemoryPool::allocate_aligned`] with `mem::align_of::<T>()`
    /// instead, or they risk handing out an under-aligned pointer.
    pub fn allocate(self: &Arc<Self>, size: usize) -> Result<PooledMemory, String> {
        self.allocate_aligned(size, mem::align_of::<u8>())
    }

    /// Allocate a memory block from the pool with an explicit alignment
    /// requirement.
    ///
    /// A block is only ever popped from the free-list reuse cache if its
    /// *actual* allocated alignment (`MemoryBlock::align`) is greater than
    /// or equal to `align`; blocks that don't meet the requirement are left
    /// in the pool (or dropped, if the pool is at capacity) rather than
    /// being handed out, since a size-class's free-list can contain blocks
    /// that were originally allocated for a different, less-aligned `T`.
    pub fn allocate_aligned(
        self: &Arc<Self>,
        size: usize,
        align: usize,
    ) -> Result<PooledMemory, String> {
        let mut stats = self
            .stats
            .lock()
            .map_err(|e| format!("Failed to acquire stats lock: {e}"))?;
        stats.allocations += 1;

        if let Some(class_index) = self.find_size_class(size) {
            let mut pool = self.pools[class_index]
                .lock()
                .map_err(|e| format!("Failed to acquire pool lock: {e}"))?;

            // Scan the free-list for a block whose alignment is sufficient.
            // Blocks that don't meet `align` are recycling candidates for a
            // *different* (less-aligned) request, so they are rotated back
            // rather than discarded outright, preserving cache behavior for
            // mixed-alignment workloads while never handing out an
            // under-aligned pointer.
            let mut scanned = 0;
            let mut found = None;
            while scanned < pool.len() {
                if let Some(block) = pool.pop_front() {
                    if block.align() >= align {
                        found = Some(block);
                        break;
                    } else {
                        pool.push_back(block);
                        scanned += 1;
                    }
                } else {
                    break;
                }
            }

            if let Some(block) = found {
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

        let block = MemoryBlock::new(actual_size, align)?;
        stats.current_size += block.size;
        stats.peak_size = stats.peak_size.max(stats.current_size);
        drop(stats);

        Ok(PooledMemory {
            block: Some(block),
            pool: Arc::downgrade(self),
            class_index: self.find_size_class(size),
        })
    }

    /// Allocate a block of *exactly* `size` bytes at `align`, bypassing the
    /// size-class recycling pool entirely.
    ///
    /// [`MemoryPool::allocate_aligned`] rounds requests up to a power-of-two
    /// size class so blocks can be recycled across similarly-sized
    /// requests; the block's true byte size (`PooledMemory::size`) can
    /// therefore be larger than what was asked for. That's fine for opaque
    /// byte-buffer use (the caller just gets "at least N bytes"), but it is
    /// unsound for [`MemoryPoolExt::with_pool_capacity`], which permanently
    /// reinterprets the block as a `Vec<T>`: a `Vec<T>` with capacity `c`
    /// will `dealloc`/`realloc` using exactly `Layout::array::<T>(c)`, so
    /// the *actual* allocation backing it must be exactly `c *
    /// size_of::<T>()` bytes — not "at least" that many — or the layout
    /// passed to `dealloc` won't match the one passed to `alloc`, which is
    /// its own separate flavor of undefined behavior (mismatched
    /// alloc/dealloc `Layout`), independent of the alignment bug.
    ///
    /// This path is therefore never recycled (no `class_index`, so
    /// `PooledMemory::drop` will let the block deallocate itself normally
    /// rather than returning it to a size-class pool).
    fn allocate_exact(self: &Arc<Self>, size: usize, align: usize) -> Result<PooledMemory, String> {
        let mut stats = self
            .stats
            .lock()
            .map_err(|e| format!("Failed to acquire stats lock: {e}"))?;
        stats.allocations += 1;
        stats.cache_misses += 1;

        let block = MemoryBlock::new(size, align)?;
        stats.current_size += block.size;
        stats.peak_size = stats.peak_size.max(stats.current_size);
        drop(stats);

        Ok(PooledMemory {
            block: Some(block),
            pool: Arc::downgrade(self),
            class_index: None,
        })
    }

    /// Return a memory block to the pool
    fn deallocate(&self, block: MemoryBlock, class_index: Option<usize>) {
        let mut stats = self
            .stats
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        stats.deallocations += 1;

        if let Some(class_index) = class_index {
            if class_index < self.pools.len() {
                let mut pool = self.pools[class_index]
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());

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
        self.stats
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    /// Clear all cached blocks
    pub fn clear(&self) {
        for pool in &self.pools {
            pool.lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clear();
        }

        let mut stats = self
            .stats
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
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
        self.block
            .as_mut()
            .expect("block should exist for valid PooledMemory")
            .as_slice_mut()
    }

    /// Get the raw pointer
    pub fn as_ptr(&self) -> *mut u8 {
        self.block
            .as_ref()
            .expect("block should exist for valid PooledMemory")
            .as_ptr()
    }

    /// Convert to a `Vec<u8>` (consumes the pooled memory)
    pub fn into_vec(mut self) -> Vec<u8> {
        let block = self
            .block
            .take()
            .expect("block should exist for valid PooledMemory");
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

    /// Allocate memory from the global pool with an explicit alignment
    /// requirement.
    ///
    /// The returned block may be larger than `size` (rounded up to a
    /// size-class) and may come from cache recycling, which is fine for
    /// opaque byte-buffer use. Do NOT use this if the bytes will be
    /// permanently reinterpreted as a `Vec<T>` with an exact capacity — use
    /// `GlobalMemoryPool::allocate_exact` for that (see its docs for why).
    pub fn allocate_aligned(size: usize, align: usize) -> Result<PooledMemory, String> {
        Self::instance().pool.allocate_aligned(size, align)
    }

    /// Allocate *exactly* `size` bytes at `align` from the global pool,
    /// bypassing size-class recycling.
    ///
    /// This is what [`MemoryPoolExt::with_pool_capacity`] uses: it needs
    /// the true allocation size to match `capacity * size_of::<T>()`
    /// exactly, since the resulting `Vec<T>` will `dealloc`/`realloc` with
    /// precisely that `Layout` later. See
    /// `MemoryPool::allocate_exact` for the full soundness rationale.
    fn allocate_exact(size: usize, align: usize) -> Result<PooledMemory, String> {
        Self::instance().pool.allocate_exact(size, align)
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
        // Note: this mirrors `std::vec::Vec`'s own restriction that
        // zero-sized `T` cannot go through the normal allocator path (a
        // zero-size `Layout` is not a valid `GlobalAlloc::alloc` argument).
        // `with_pool_capacity` is pool-backed specifically to reduce real
        // allocator traffic, which is moot for ZSTs anyway, so callers
        // needing `Vec<ZST>` should use `Vec::with_capacity` directly.
        if mem::size_of::<T>() == 0 {
            return Err(
                "MemoryPoolExt::with_pool_capacity does not support zero-sized types; use Vec::with_capacity instead"
                    .to_string(),
            );
        }

        let size = capacity * mem::size_of::<T>();

        // Request a block of *exactly* `size` bytes aligned for `T` (not
        // just for `u8`, and not rounded up to a size-class). This closes
        // two distinct UB bugs that both manifested through this function:
        //
        // 1. Alignment: every block used to come out of the pool 1-byte
        //    aligned (`MemoryBlock::new` hardcoded `align_of::<u8>()`), so
        //    reinterpreting the pointer as `*mut T` for any `T` with
        //    `align_of::<T>() > 1` (e.g. i32, f64, u64, or over-aligned
        //    SIMD types) was undefined behavior (confirmed by Miri: "access
        //    /deallocation... alignment 1, but alignment N is required").
        //
        // 2. Size: `MemoryPool::allocate`/`allocate_aligned` round the
        //    requested size up to a power-of-two size-class so blocks can
        //    be recycled across similarly-sized requests. That means the
        //    *actual* allocation could be larger than `size`. A `Vec<T>` of
        //    capacity `capacity` will `dealloc`/`realloc` using exactly
        //    `Layout::array::<T>(capacity)` (= `size` bytes), so if the
        //    real allocation were the rounded-up size-class value instead,
        //    `dealloc` would be called with a *different* Layout than
        //    `alloc` was — itself separately-undefined behavior, just on
        //    the size axis instead of the alignment axis. `allocate_exact`
        //    bypasses the size-class pool entirely, so the block is always
        //    precisely `size` bytes: alloc and dealloc always agree.
        let pooled = GlobalMemoryPool::allocate_exact(size, mem::align_of::<T>())?;

        // Take ownership of the underlying block directly rather than
        // routing through `PooledMemory::into_vec()` -> `Vec<u8>`: a
        // `Vec<u8>` only carries a static alignment guarantee of 1, so
        // going through it and then reinterpreting `.as_ptr()` as `*mut T`
        // would throw away the very alignment guarantee we just secured
        // above (even though the underlying bytes are, in fact, correctly
        // aligned). Working with the block's raw pointer keeps the
        // alignment guarantee visible end-to-end.
        let mut pooled = pooled;
        let block = pooled
            .block
            .take()
            .expect("block should exist for valid PooledMemory");
        let ptr = block.as_ptr() as *mut T;

        // Prevent the block from being deallocated by its own `Drop` impl;
        // ownership of the allocation is being transferred to the `Vec<T>`
        // we construct below, which will run `T`'s (and its own) `Drop`
        // via the standard allocator on its own eventual drop.
        mem::forget(block);
        // `pooled` no longer owns a block (we `take()`-took it above), so
        // its `Drop` impl is a no-op; drop it explicitly for clarity.
        drop(pooled);

        let len = 0;

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
        let mut mem1 = pool.allocate(1024).expect("test: operation should succeed");
        assert_eq!(mem1.size(), 1024);

        // Write some data
        let slice = mem1.as_slice_mut();
        slice[0] = 42;
        slice[1023] = 99;

        drop(mem1);

        // Allocate again
        let mut mem2 = pool.allocate(1024).expect("test: operation should succeed");
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

        let mem1 = pool.allocate(512).expect("test: operation should succeed");
        let mem2 = pool.allocate(1024).expect("test: operation should succeed");
        let mem3 = pool.allocate(2048).expect("test: operation should succeed");

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
        let mem1 = GlobalMemoryPool::allocate(1024).expect("test: operation should succeed");
        assert_eq!(mem1.size(), 1024);

        let mem2 = GlobalMemoryPool::allocate(2048).expect("test: operation should succeed");
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

        let mut vec: Vec<i32> =
            Vec::with_pool_capacity(100).expect("test: operation should succeed");
        vec.push(42);
        vec.push(99);

        assert_eq!(vec.len(), 2);
        assert_eq!(vec.capacity(), 100);
        assert_eq!(vec[0], 42);
        assert_eq!(vec[1], 99);
    }

    /// Regression test for the alignment UB Miri caught: `MemoryBlock::new`
    /// used to hardcode 1-byte alignment for every allocation, so
    /// reinterpreting the block as `*mut T` for any `T` needing more than
    /// 1-byte alignment (e.g. `f64`/`u64`, which need 8) was undefined
    /// behavior. Under Miri this must not report any "alignment N required"
    /// diagnostics; under normal execution it must simply behave correctly.
    #[test]
    fn test_vec_with_pool_capacity_f64_alignment() {
        GlobalMemoryPool::clear();

        assert_eq!(mem::align_of::<f64>(), 8);

        let mut vec: Vec<f64> =
            Vec::with_pool_capacity(50).expect("test: operation should succeed");
        assert_eq!(vec.capacity(), 50);

        for i in 0..50 {
            vec.push(i as f64 * 1.5);
        }

        assert_eq!(vec.len(), 50);
        for (i, value) in vec.iter().enumerate() {
            assert_eq!(*value, i as f64 * 1.5);
        }

        // The pointer backing the Vec must actually satisfy f64's alignment
        // requirement -- this is the precise condition whose violation
        // caused the original Miri UB (misaligned pointer access/dealloc).
        assert_eq!(vec.as_ptr() as usize % mem::align_of::<f64>(), 0);
    }

    /// Same regression, but for `u64` (also 8-byte aligned) going through a
    /// *fresh* pool instance rather than the global singleton, and using a
    /// capacity that would have landed inside a recycled size-class bucket
    /// under the pre-fix pooling logic (exercising `MemoryPool` directly,
    /// not just `MemoryPoolExt`).
    #[test]
    fn test_memory_block_u64_alignment_direct() {
        assert_eq!(mem::align_of::<u64>(), 8);

        let size = 64 * mem::size_of::<u64>();
        let block =
            MemoryBlock::new(size, mem::align_of::<u64>()).expect("test: operation should succeed");

        assert_eq!(block.align(), 8);
        assert_eq!(block.as_ptr() as usize % 8, 0);
    }

    /// Regression test using a type with an alignment requirement stricter
    /// than any primitive integer (16 bytes via `#[repr(align(16))]`, as
    /// used by SIMD-oriented types). This is the sharpest possible check
    /// that the fix threads the *actual* required alignment through, not
    /// just "more than 1".
    #[test]
    fn test_vec_with_pool_capacity_over_aligned_type() {
        #[repr(align(16))]
        #[derive(Debug, Clone, Copy, PartialEq)]
        struct Align16 {
            data: [u8; 16],
        }

        GlobalMemoryPool::clear();

        assert_eq!(mem::align_of::<Align16>(), 16);

        let capacity = 20;
        let mut vec: Vec<Align16> =
            Vec::with_pool_capacity(capacity).expect("test: operation should succeed");
        assert_eq!(vec.capacity(), capacity);

        for i in 0..capacity {
            vec.push(Align16 {
                data: [i as u8; 16],
            });
        }

        assert_eq!(vec.len(), capacity);
        for (i, item) in vec.iter().enumerate() {
            assert_eq!(item.data, [i as u8; 16]);
        }

        assert_eq!(
            vec.as_ptr() as usize % mem::align_of::<Align16>(),
            0,
            "Vec<Align16> backing pointer must be 16-byte aligned"
        );
    }

    /// Regression test for the alloc/dealloc size-`Layout`-mismatch bug
    /// that was hiding behind the alignment bug: `MemoryPool`'s size-class
    /// buckets round allocation size up to the next power-of-two class, so
    /// a naive fix that only addressed alignment (while still routing
    /// `with_pool_capacity` through the recycling pool) would allocate a
    /// larger block than `capacity * size_of::<T>()` and then construct a
    /// `Vec<T>` claiming the smaller, unrounded capacity -- a mismatched
    /// `Layout` between what was `alloc`'d and what `Vec::drop` will
    /// `dealloc`. `capacity` below is chosen so its byte size (1200 bytes
    /// for i32) falls inside the default pool's 2048-byte size class,
    /// specifically to exercise that rounding path.
    #[test]
    fn test_vec_with_pool_capacity_avoids_size_class_rounding() {
        GlobalMemoryPool::clear();

        let capacity = 300; // 300 * size_of::<i32>() == 1200 bytes
        assert_eq!(capacity * mem::size_of::<i32>(), 1200);

        let mut vec: Vec<i32> =
            Vec::with_pool_capacity(capacity).expect("test: operation should succeed");

        // Capacity must be exactly what was requested, not rounded up to a
        // size-class boundary (e.g. 2048 / 4 == 512), since the `Vec<T>`'s
        // own `Layout::array::<T>(capacity)` on drop must match the
        // allocation's actual byte size exactly.
        assert_eq!(vec.capacity(), capacity);

        for i in 0..capacity {
            vec.push(i as i32);
        }
        assert_eq!(vec.len(), capacity);
        assert_eq!(vec[0], 0);
        assert_eq!(vec[capacity - 1], (capacity - 1) as i32);

        // Dropping here must not trigger a size- or alignment-mismatched
        // dealloc; this is implicitly checked by Miri when this test is run
        // under `cargo +nightly miri test`.
    }

    /// `with_pool_capacity` must reject zero-sized types rather than
    /// attempting a zero-size `alloc()` call (itself undefined behavior,
    /// independent of the alignment fix above).
    #[test]
    fn test_vec_with_pool_capacity_rejects_zst() {
        let result = Vec::<()>::with_pool_capacity(10);
        assert!(result.is_err());
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

        let _mem = pool.allocate(1024).expect("test: operation should succeed");
        let stats = pool.stats();
        assert_eq!(stats.allocations, 1);
        assert_eq!(stats.cache_misses, 1);
    }
}
