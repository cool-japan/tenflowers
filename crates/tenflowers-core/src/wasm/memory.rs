//! WASM memory management: WasmAllocator for managing heap allocations within WASM constraints.

#[allow(unused_imports)]
use crate::{Result, TensorError};
#[allow(unused_imports)]
use std::collections::HashMap;

/// WASM-specific memory management
#[cfg(target_arch = "wasm32")]
pub struct WasmAllocator {
    allocated: std::cell::RefCell<HashMap<*mut u8, usize>>,
    total_allocated: std::cell::RefCell<usize>,
    memory_limit: usize,
}

#[cfg(target_arch = "wasm32")]
impl WasmAllocator {
    pub fn new(memory_limit: usize) -> Self {
        Self {
            allocated: std::cell::RefCell::new(HashMap::new()),
            total_allocated: std::cell::RefCell::new(0),
            memory_limit,
        }
    }

    pub fn allocate(&self, size: usize) -> Result<*mut u8> {
        let current_total = *self.total_allocated.borrow();
        if current_total + size > self.memory_limit {
            return Err(TensorError::allocation_error_simple(&format!(
                "Would exceed memory limit: {} + {} > {}",
                current_total, size, self.memory_limit
            )));
        }

        if size == 0 {
            return Ok(std::ptr::null_mut());
        }

        let layout = std::alloc::Layout::from_size_align(size, 32)
            .map_err(|e| TensorError::allocation_error_simple(e.to_string()))?;

        let ptr = unsafe { std::alloc::alloc(layout) };
        if ptr.is_null() {
            return Err(TensorError::allocation_error_simple(&format!(
                "Failed to allocate {} bytes",
                size
            )));
        }

        self.allocated.borrow_mut().insert(ptr, size);
        *self.total_allocated.borrow_mut() += size;

        Ok(ptr)
    }

    pub unsafe fn deallocate(&self, ptr: *mut u8) -> Result<()> {
        if ptr.is_null() {
            return Ok(());
        }

        if let Some(size) = self.allocated.borrow_mut().remove(&ptr) {
            let layout = std::alloc::Layout::from_size_align_unchecked(size, 32);
            std::alloc::dealloc(ptr, layout);
            *self.total_allocated.borrow_mut() -= size;
        }

        Ok(())
    }

    pub fn total_allocated(&self) -> usize {
        *self.total_allocated.borrow()
    }

    pub fn memory_limit(&self) -> usize {
        self.memory_limit
    }
}
