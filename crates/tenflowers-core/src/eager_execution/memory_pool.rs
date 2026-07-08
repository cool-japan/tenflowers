//! Memory pool implementation for eager execution

use crate::device::context::DEVICE_MANAGER;
use crate::{Device, Result};
use std::collections::HashMap;
use std::sync::RwLock;
use std::time::{Duration, Instant};

use super::config::EagerExecutionConfig;

/// Memory pool for fast allocation/deallocation
#[allow(dead_code)]
pub(super) struct MemoryPool {
    pub(super) blocks: RwLock<HashMap<Device, Vec<MemoryBlock>>>,
    pub(super) config: EagerExecutionConfig,
}

#[derive(Debug)]
#[allow(dead_code)]
pub(super) struct MemoryBlock {
    pub(super) ptr: *mut u8,
    pub(super) size: usize,
    pub(super) available: bool,
    pub(super) last_used: Instant,
}

unsafe impl Send for MemoryBlock {}
unsafe impl Sync for MemoryBlock {}

#[allow(dead_code)]
impl MemoryPool {
    pub(super) fn new(config: EagerExecutionConfig) -> Self {
        Self {
            blocks: RwLock::new(HashMap::new()),
            config,
        }
    }

    pub(super) fn allocate(&self, device: &Device, size: usize) -> Result<*mut u8> {
        let _start = Instant::now();

        // Try to find available block
        {
            let mut blocks = self.blocks.write().map_err(|_| {
                crate::TensorError::invalid_operation_simple(
                    "memory pool lock poisoned".to_string(),
                )
            })?;
            let device_blocks = blocks.entry(*device).or_default();

            for block in device_blocks.iter_mut() {
                if block.available && block.size >= size {
                    block.available = false;
                    block.last_used = Instant::now();
                    return Ok(block.ptr);
                }
            }
        }

        // Allocate new block if none available
        let context = DEVICE_MANAGER.get_context(device)?;
        let ptr = context.allocator().allocate(size)?;

        // Add to pool
        {
            let mut blocks = self.blocks.write().map_err(|_| {
                crate::TensorError::invalid_operation_simple(
                    "memory pool lock poisoned".to_string(),
                )
            })?;
            let device_blocks = blocks.entry(*device).or_default();
            device_blocks.push(MemoryBlock {
                ptr,
                size,
                available: false,
                last_used: Instant::now(),
            });
        }

        Ok(ptr)
    }

    pub(super) fn deallocate(&self, device: &Device, ptr: *mut u8) -> Result<()> {
        let mut blocks = self.blocks.write().map_err(|_| {
            crate::TensorError::invalid_operation_simple("memory pool lock poisoned".to_string())
        })?;
        if let Some(device_blocks) = blocks.get_mut(device) {
            for block in device_blocks.iter_mut() {
                if block.ptr == ptr {
                    block.available = true;
                    block.last_used = Instant::now();
                    return Ok(());
                }
            }
        }
        Ok(())
    }

    pub(super) fn cleanup_old_blocks(&self) {
        let threshold = Duration::from_secs(60); // 1 minute
        let now = Instant::now();

        let mut blocks = self
            .blocks
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        for device_blocks in blocks.values_mut() {
            device_blocks.retain(|block| {
                if block.available && now.duration_since(block.last_used) > threshold {
                    // Deallocate old unused blocks
                    false
                } else {
                    true
                }
            });
        }
    }

    /// Pre-warm memory pool by pre-allocating blocks of the required size
    pub(super) fn pre_warm(&self, device: &Device, size: usize, num_blocks: usize) -> Result<()> {
        let context = DEVICE_MANAGER.get_context(device)?;
        let mut blocks = self.blocks.write().map_err(|_| {
            crate::TensorError::invalid_operation_simple("memory pool lock poisoned".to_string())
        })?;
        let device_blocks = blocks.entry(*device).or_default();

        // Pre-allocate the specified number of blocks
        for _ in 0..num_blocks {
            let ptr = context.allocator().allocate(size)?;
            device_blocks.push(MemoryBlock {
                ptr,
                size,
                available: true, // Available for use
                last_used: Instant::now(),
            });
        }

        Ok(())
    }
}

/// Memory guard for RAII memory management
#[allow(dead_code)]
pub(super) struct MemoryGuard {
    pub(super) device: Device,
    pub(super) estimated_memory: usize,
    pub(super) operation: String,
}
