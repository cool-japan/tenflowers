//! Memory pool management for efficient GPU memory allocation
//!
//! This module provides memory pool allocators with reference counting,
//! block management, and automatic defragmentation capabilities.

use crate::{Device, Result, TensorError};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

/// Memory pool statistics for monitoring
#[derive(Debug, Clone)]
pub struct MemoryPoolStats {
    pub total_allocated: usize,
    pub total_free: usize,
    pub blocks_allocated: usize,
    pub blocks_free: usize,
    pub fragmentation_ratio: f32,
    pub peak_allocated: usize,
    pub allocation_count: u64,
    pub deallocation_count: u64,
    pub defragmentation_count: u64,
    pub largest_free_block: usize,
    pub average_block_size: f32,
    pub memory_pressure: f32,
}

/// Allocation tracking for analytics
#[derive(Debug, Clone)]
pub struct AllocationTracker {
    pub timestamp: Instant,
    pub size: usize,
    pub block_idx: usize,
    pub lifetime_us: Option<u64>,
    pub deallocated_at: Option<Instant>,
}

/// Memory pressure levels
#[derive(Debug, Clone, PartialEq)]
pub enum MemoryPressureLevel {
    Low,      // < 50% usage
    Medium,   // 50-80% usage
    High,     // 80-95% usage
    Critical, // > 95% usage
}

/// Memory block metadata
#[derive(Debug, Clone)]
pub(crate) struct MemoryBlock {
    #[allow(dead_code)] // Used in GPU-feature-gated methods
    pub offset: usize,
    pub size: usize,
    pub is_free: bool,
    #[allow(dead_code)] // Used in GPU-feature-gated methods
    pub ref_count: usize, // Reference count for shared buffer management
}

impl MemoryBlock {
    /// Create a new free memory block
    #[allow(dead_code)] // Used in GPU-feature-gated methods
    pub fn new_free(offset: usize, size: usize) -> Self {
        Self {
            offset,
            size,
            is_free: true,
            ref_count: 0,
        }
    }

    /// Create a new allocated memory block with initial reference count
    #[allow(dead_code)] // Used in GPU-feature-gated methods
    pub fn new_allocated(offset: usize, size: usize) -> Self {
        Self {
            offset,
            size,
            is_free: false,
            ref_count: 1, // Start with one reference
        }
    }

    /// Increment reference count (for shared buffer access)
    #[allow(dead_code)] // Used in GPU-feature-gated methods
    pub fn add_ref(&mut self) {
        assert!(!self.is_free, "Cannot add reference to free block");
        self.ref_count += 1;
    }

    /// Decrement reference count and return true if should be freed
    #[allow(dead_code)] // Used in GPU-feature-gated methods
    pub fn release_ref(&mut self) -> bool {
        assert!(!self.is_free, "Cannot release reference from free block");
        assert!(self.ref_count > 0, "Reference count underflow");

        self.ref_count -= 1;
        self.ref_count == 0 // Return true if no more references
    }

    /// Check if block can be freed (no references remaining)
    #[allow(dead_code)] // Used in GPU-feature-gated methods
    pub fn can_free(&self) -> bool {
        !self.is_free && self.ref_count == 0
    }
}

/// Memory pool allocator for efficient GPU memory management
#[derive(Debug)]
pub struct MemoryPool {
    #[allow(dead_code)]
    device: Device,
    #[cfg(feature = "gpu")]
    gpu_device: Arc<wgpu::Device>,
    #[cfg(feature = "gpu")]
    gpu_queue: Arc<wgpu::Queue>,

    // Memory pool data
    #[allow(dead_code)]
    pool_size: usize,
    #[cfg(feature = "gpu")]
    pool_buffer: wgpu::Buffer,

    // Block management
    #[allow(dead_code)]
    blocks: Arc<RwLock<Vec<MemoryBlock>>>,
    #[allow(dead_code)]
    free_blocks: Arc<Mutex<VecDeque<usize>>>, // Indices of free blocks

    // Statistics and analytics
    stats: Arc<Mutex<MemoryPoolStats>>,
    #[allow(dead_code)]
    allocation_history: Arc<Mutex<HashMap<usize, AllocationTracker>>>,

    // Defragmentation settings
    #[allow(dead_code)]
    auto_defrag_threshold: f32, // Trigger defragmentation when fragmentation > threshold
    #[allow(dead_code)]
    defrag_last_run: Arc<Mutex<Instant>>,
    #[allow(dead_code)]
    defrag_min_interval: Duration, // Minimum time between defragmentation runs
}

impl MemoryPool {
    /// Create a new memory pool with specified size in bytes
    #[cfg(feature = "gpu")]
    pub fn new(device_id: usize, pool_size: usize) -> Result<Self> {
        let gpu_ctx = crate::device::context::get_gpu_context(device_id)?;

        // Create large buffer for memory pool
        let pool_buffer = gpu_ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("memory_pool_buffer"),
            size: pool_size as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Initialize with single large free block
        let blocks = vec![MemoryBlock::new_free(0, pool_size)];

        let mut free_blocks = VecDeque::new();
        free_blocks.push_back(0);

        let stats = MemoryPoolStats {
            total_allocated: 0,
            total_free: pool_size,
            blocks_allocated: 0,
            blocks_free: 1,
            fragmentation_ratio: 0.0,
            peak_allocated: 0,
            allocation_count: 0,
            deallocation_count: 0,
            defragmentation_count: 0,
            largest_free_block: pool_size,
            average_block_size: pool_size as f32,
            memory_pressure: 0.0,
        };

        Ok(Self {
            device: Device::Gpu(device_id),
            gpu_device: gpu_ctx.device.clone(),
            gpu_queue: gpu_ctx.queue.clone(),
            pool_size,
            pool_buffer,
            blocks: Arc::new(RwLock::new(blocks)),
            free_blocks: Arc::new(Mutex::new(free_blocks)),
            stats: Arc::new(Mutex::new(stats)),
            allocation_history: Arc::new(Mutex::new(HashMap::new())),
            auto_defrag_threshold: 0.25, // Auto-defrag when 25% fragmented
            defrag_last_run: Arc::new(Mutex::new(Instant::now())),
            defrag_min_interval: Duration::from_secs(30), // Min 30 seconds between defrags
        })
    }

    /// Allocate memory from the pool
    #[cfg(feature = "gpu")]
    pub fn allocate(&self, size: usize, alignment: usize) -> Result<PooledBuffer<'_>> {
        let aligned_size = align_size(size, alignment);

        let mut free_blocks = self.free_blocks.lock().unwrap();
        let mut blocks = self.blocks.write().unwrap();

        // Find suitable free block using best-fit strategy
        let mut best_block_idx = None;
        let mut best_size = usize::MAX;

        for &block_idx in free_blocks.iter() {
            let block = &blocks[block_idx];
            if block.is_free && block.size >= aligned_size && block.size < best_size {
                best_block_idx = Some(block_idx);
                best_size = block.size;
            }
        }

        if let Some(block_idx) = best_block_idx {
            // Get information from the block before splitting
            let (offset, block_size) = {
                let block = &blocks[block_idx];
                (block.offset, block.size)
            };

            // Split block if necessary
            if block_size > aligned_size {
                // Create new free block for remainder
                let new_block =
                    MemoryBlock::new_free(offset + aligned_size, block_size - aligned_size);
                blocks.push(new_block);
                free_blocks.push_back(blocks.len() - 1);
            }

            // Mark block as allocated using new constructor logic
            blocks[block_idx] = MemoryBlock::new_allocated(offset, aligned_size);

            // Remove from free blocks
            free_blocks.retain(|&idx| idx != block_idx);

            // Track allocation for analytics
            let mut history = self.allocation_history.lock().unwrap();
            history.insert(
                block_idx,
                AllocationTracker {
                    timestamp: Instant::now(),
                    size: aligned_size,
                    block_idx,
                    lifetime_us: None,
                    deallocated_at: None,
                },
            );

            // Update statistics
            self.update_enhanced_stats(&blocks);

            // Check if auto-defragmentation should be triggered
            #[cfg(feature = "gpu")]
            self.maybe_auto_defrag();

            Ok(PooledBuffer {
                pool: self,
                block_idx,
                offset,
                size: aligned_size,
            })
        } else {
            Err(TensorError::allocation_error_simple(format!(
                "Cannot allocate {} bytes from memory pool",
                aligned_size
            )))
        }
    }

    /// Deallocate memory back to the pool (reference counting aware)
    #[cfg(feature = "gpu")]
    pub(crate) fn deallocate(&self, block_idx: usize) -> Result<()> {
        let mut blocks = self.blocks.write().unwrap();
        let mut free_blocks = self.free_blocks.lock().unwrap();

        let block = &mut blocks[block_idx];
        if block.is_free {
            return Err(TensorError::invalid_argument(
                "Attempting to deallocate already free block".to_string(),
            ));
        }

        // Decrement reference count and only free if no references remain
        if block.release_ref() {
            // No more references, free the block
            block.is_free = true;
            free_blocks.push_back(block_idx);
        }

        // Update allocation tracking with lifetime
        let mut history = self.allocation_history.lock().unwrap();
        if let Some(_tracker) = history.remove(&block_idx) {
            // Tracker removed from history - could store in a completed allocations log for further analysis
        }

        // Coalesce adjacent free blocks to reduce fragmentation
        self.coalesce_blocks(&mut blocks, &mut free_blocks);

        // Update statistics
        self.update_enhanced_stats(&blocks);

        Ok(())
    }

    /// Share a buffer by incrementing its reference count
    /// Returns true if the buffer was successfully shared
    #[cfg(feature = "gpu")]
    pub fn share_buffer(&self, block_idx: usize) -> Result<bool> {
        let mut blocks = self.blocks.write().unwrap();

        if block_idx >= blocks.len() {
            return Err(TensorError::invalid_argument(format!(
                "Invalid block index: {}",
                block_idx
            )));
        }

        let block = &mut blocks[block_idx];
        if block.is_free {
            return Err(TensorError::invalid_argument(
                "Cannot share a free block".to_string(),
            ));
        }

        block.add_ref();
        Ok(true)
    }

    /// Release a reference to a shared buffer
    /// Returns true if the buffer was actually freed (reference count reached 0)
    #[cfg(feature = "gpu")]
    pub fn release_buffer(&self, block_idx: usize) -> Result<bool> {
        let mut blocks = self.blocks.write().unwrap();
        let mut free_blocks = self.free_blocks.lock().unwrap();

        if block_idx >= blocks.len() {
            return Err(TensorError::invalid_argument(format!(
                "Invalid block index: {}",
                block_idx
            )));
        }

        let block = &mut blocks[block_idx];
        if block.is_free {
            return Err(TensorError::invalid_argument(
                "Cannot release reference to already free block".to_string(),
            ));
        }

        if block.release_ref() {
            // No more references, free the block
            block.is_free = true;
            free_blocks.push_back(block_idx);

            // Update allocation tracking
            let mut history = self.allocation_history.lock().unwrap();
            if let Some(_tracker) = history.remove(&block_idx) {
                // Tracker removed from history - could store in a separate history if needed for analysis
            }

            // Update statistics
            self.update_enhanced_stats(&blocks);

            Ok(true) // Buffer was freed
        } else {
            Ok(false) // Buffer still has references
        }
    }

    /// Get the current reference count for a buffer
    #[cfg(feature = "gpu")]
    pub fn get_buffer_ref_count(&self, block_idx: usize) -> Result<usize> {
        let blocks = self.blocks.read().unwrap();

        if block_idx >= blocks.len() {
            return Err(TensorError::invalid_argument(format!(
                "Invalid block index: {}",
                block_idx
            )));
        }

        let block = &blocks[block_idx];
        if block.is_free {
            Ok(0)
        } else {
            Ok(block.ref_count)
        }
    }

    /// Coalesce adjacent free blocks to reduce fragmentation
    #[cfg(feature = "gpu")]
    fn coalesce_blocks(&self, blocks: &mut [MemoryBlock], free_blocks: &mut VecDeque<usize>) {
        // Sort free blocks by offset
        let mut free_indices: Vec<_> = free_blocks.iter().copied().collect();
        free_indices.sort_by_key(|&idx| blocks[idx].offset);

        let mut coalesced = Vec::new();
        let mut i = 0;

        while i < free_indices.len() {
            let mut current_idx = free_indices[i];
            let mut current_block = blocks[current_idx].clone();

            // Look for adjacent blocks to coalesce
            while i + 1 < free_indices.len() {
                let next_idx = free_indices[i + 1];
                let next_block = &blocks[next_idx];

                // Check if blocks are adjacent
                if current_block.offset + current_block.size == next_block.offset {
                    // Coalesce blocks
                    current_block.size += next_block.size;
                    i += 1; // Skip next block as it's now coalesced
                } else {
                    break;
                }
            }

            // Update the block
            blocks[current_idx] = current_block;
            coalesced.push(current_idx);
            i += 1;
        }

        // Update free blocks queue
        free_blocks.clear();
        for idx in coalesced {
            free_blocks.push_back(idx);
        }
    }

    /// Enhanced statistics update with advanced analytics
    #[allow(dead_code)]
    fn update_enhanced_stats(&self, blocks: &[MemoryBlock]) {
        let mut stats = self.stats.lock().unwrap();
        stats.blocks_allocated = 0;
        stats.blocks_free = 0;
        stats.total_allocated = 0;
        stats.total_free = 0;
        stats.largest_free_block = 0;

        let mut block_sizes = Vec::new();

        for block in blocks {
            block_sizes.push(block.size);
            if block.is_free {
                stats.blocks_free += 1;
                stats.total_free += block.size;
                stats.largest_free_block = stats.largest_free_block.max(block.size);
            } else {
                stats.blocks_allocated += 1;
                stats.total_allocated += block.size;
            }
        }

        // Update peak allocated
        stats.peak_allocated = stats.peak_allocated.max(stats.total_allocated);

        // Update counters
        stats.allocation_count += 1;

        // Calculate fragmentation ratio
        if stats.total_free > 0 {
            stats.fragmentation_ratio =
                stats.blocks_free as f32 / (stats.total_free as f32 / 1024.0);
        } else {
            stats.fragmentation_ratio = 0.0;
        }

        // Calculate average block size
        if !block_sizes.is_empty() {
            stats.average_block_size =
                block_sizes.iter().sum::<usize>() as f32 / block_sizes.len() as f32;
        }

        // Calculate memory pressure
        let usage_ratio = stats.total_allocated as f32 / self.pool_size as f32;
        stats.memory_pressure = usage_ratio;
    }

    /// Check if auto-defragmentation should be triggered
    #[cfg(feature = "gpu")]
    #[allow(dead_code)]
    fn maybe_auto_defrag(&self) {
        let stats = self.stats.lock().unwrap();
        if stats.fragmentation_ratio > self.auto_defrag_threshold {
            let mut last_run = self.defrag_last_run.lock().unwrap();
            if last_run.elapsed() >= self.defrag_min_interval {
                drop(stats); // Release lock before defragmentation
                self.defragment();
                *last_run = Instant::now();
            }
        }
    }

    /// Perform active defragmentation of memory pool
    #[cfg(feature = "gpu")]
    #[allow(dead_code)]
    pub fn defragment(&self) {
        let mut blocks = self.blocks.write().unwrap();
        let mut free_blocks = self.free_blocks.lock().unwrap();

        // Sort blocks by offset to enable merging
        blocks.sort_by_key(|block| block.offset);

        // Rebuild free blocks list based on sorted blocks
        free_blocks.clear();
        for (idx, block) in blocks.iter().enumerate() {
            if block.is_free {
                free_blocks.push_back(idx);
            }
        }

        // Coalesce adjacent free blocks
        self.coalesce_blocks(&mut blocks, &mut free_blocks);

        // Update statistics
        self.update_enhanced_stats(&blocks);
        let mut stats = self.stats.lock().unwrap();
        stats.defragmentation_count += 1;
    }

    /// Get current memory pressure level
    #[allow(dead_code)]
    pub fn memory_pressure_level(&self) -> MemoryPressureLevel {
        let stats = self.stats.lock().unwrap();
        match stats.memory_pressure {
            p if p < 0.5 => MemoryPressureLevel::Low,
            p if p < 0.8 => MemoryPressureLevel::Medium,
            p if p < 0.95 => MemoryPressureLevel::High,
            _ => MemoryPressureLevel::Critical,
        }
    }

    /// Force cleanup of small free blocks (aggressive defragmentation)
    #[cfg(feature = "gpu")]
    #[allow(dead_code)]
    pub fn aggressive_cleanup(&self, min_block_size: usize) -> Result<usize> {
        let mut blocks = self.blocks.write().unwrap();
        let mut free_blocks = self.free_blocks.lock().unwrap();

        let mut removed_count = 0;

        // Remove small free blocks and merge their space
        let mut i = 0;
        while i < blocks.len() {
            if blocks[i].is_free && blocks[i].size < min_block_size {
                blocks.remove(i);
                removed_count += 1;
            } else {
                i += 1;
            }
        }

        // Rebuild free blocks list
        free_blocks.clear();
        for (idx, block) in blocks.iter().enumerate() {
            if block.is_free {
                free_blocks.push_back(idx);
            }
        }

        // Coalesce remaining blocks
        self.coalesce_blocks(&mut blocks, &mut free_blocks);

        // Update statistics
        self.update_enhanced_stats(&blocks);

        Ok(removed_count)
    }

    /// Get memory pool statistics
    pub fn stats(&self) -> MemoryPoolStats {
        self.stats.lock().unwrap().clone()
    }

    /// Get pool buffer reference
    #[cfg(feature = "gpu")]
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.pool_buffer
    }

    /// Get GPU device reference
    #[cfg(feature = "gpu")]
    pub fn device(&self) -> &wgpu::Device {
        &self.gpu_device
    }

    /// Get GPU queue reference
    #[cfg(feature = "gpu")]
    pub fn queue(&self) -> &wgpu::Queue {
        &self.gpu_queue
    }
}

/// A buffer allocated from the memory pool
#[derive(Debug)]
pub struct PooledBuffer<'a> {
    #[allow(dead_code)]
    pool: &'a MemoryPool,
    #[allow(dead_code)]
    block_idx: usize,
    offset: usize,
    size: usize,
}

impl<'a> PooledBuffer<'a> {
    /// Get the offset within the pool buffer
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Get the size of the allocated buffer
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get reference to the underlying pool buffer
    #[cfg(feature = "gpu")]
    pub fn buffer(&self) -> &wgpu::Buffer {
        self.pool.buffer()
    }

    /// Create a view of this buffer with offset and size
    pub fn view(&'a self, offset: usize, size: usize) -> Result<BufferView<'a>> {
        if offset + size > self.size {
            return Err(TensorError::invalid_argument(format!(
                "View out of bounds: offset={}, size={}, buffer_size={}",
                offset, size, self.size
            )));
        }

        Ok(BufferView {
            buffer: self,
            view_offset: offset,
            view_size: size,
        })
    }
}

#[cfg(feature = "gpu")]
impl<'a> Drop for PooledBuffer<'a> {
    fn drop(&mut self) {
        // Deallocate when buffer is dropped
        let _ = self.pool.deallocate(self.block_idx);
    }
}

/// A view into a pooled buffer for zero-copy operations
pub struct BufferView<'a> {
    buffer: &'a PooledBuffer<'a>,
    view_offset: usize,
    view_size: usize,
}

impl<'a> BufferView<'a> {
    /// Get the absolute offset within the pool buffer
    pub fn absolute_offset(&self) -> usize {
        self.buffer.offset() + self.view_offset
    }

    /// Get the size of the view
    pub fn size(&self) -> usize {
        self.view_size
    }

    /// Get reference to the underlying pool buffer
    #[cfg(feature = "gpu")]
    pub fn buffer(&self) -> &wgpu::Buffer {
        self.buffer.buffer()
    }
}

/// Utility function to align size to boundary
#[allow(dead_code)]
pub fn align_size(size: usize, alignment: usize) -> usize {
    (size + alignment - 1) & !(alignment - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_align_size() {
        assert_eq!(align_size(13, 8), 16);
        assert_eq!(align_size(16, 8), 16);
        assert_eq!(align_size(17, 8), 24);
    }

    #[test]
    fn test_memory_block() {
        let block = MemoryBlock::new_free(0, 1024);
        assert!(block.is_free);
        assert_eq!(block.size, 1024);
        assert_eq!(block.ref_count, 0);

        let mut allocated_block = MemoryBlock::new_allocated(1024, 512);
        assert!(!allocated_block.is_free);
        assert_eq!(allocated_block.ref_count, 1);

        allocated_block.add_ref();
        assert_eq!(allocated_block.ref_count, 2);

        assert!(!allocated_block.release_ref());
        assert_eq!(allocated_block.ref_count, 1);

        assert!(allocated_block.release_ref());
        assert_eq!(allocated_block.ref_count, 0);
    }

    #[test]
    fn test_memory_pressure_level() {
        let pressure = MemoryPressureLevel::Low;
        assert_eq!(pressure, MemoryPressureLevel::Low);

        let high_pressure = MemoryPressureLevel::High;
        assert_eq!(high_pressure, MemoryPressureLevel::High);
    }
}
