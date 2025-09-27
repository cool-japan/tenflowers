#![allow(clippy::result_large_err)]

use crate::{Device, Result, TensorError};
use std::sync::Arc;

/// Trait for tensor storage backends
pub trait TensorBuffer: Send + Sync {
    type Elem: Clone;

    /// Get the device where this buffer is allocated
    fn device(&self) -> &Device;

    /// Get the number of elements in the buffer
    fn len(&self) -> usize;

    /// Check if the buffer is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the size in bytes
    fn size_bytes(&self) -> usize;

    /// Clone the buffer (may involve device memory copy)
    fn clone_buffer(&self) -> Result<Box<dyn TensorBuffer<Elem = Self::Elem>>>;

    /// Create a view into this buffer (zero-copy when possible)
    fn view(&self, offset: usize, len: usize) -> Result<Box<dyn TensorBuffer<Elem = Self::Elem>>>;

    /// Convert to a CPU buffer for host operations
    fn to_cpu(&self) -> Result<Vec<Self::Elem>>;

    /// Get raw pointer (unsafe - for FFI and low-level ops)
    ///
    /// # Safety
    /// The caller must ensure the pointer is valid for the lifetime of the buffer
    /// and that no mutable references exist when using this pointer.
    unsafe fn as_ptr(&self) -> *const Self::Elem;

    /// Get mutable raw pointer (unsafe - for FFI and low-level ops)
    ///
    /// # Safety
    /// The caller must ensure the pointer is valid for the lifetime of the buffer
    /// and that no other references (mutable or immutable) exist when using this pointer.
    unsafe fn as_mut_ptr(&mut self) -> *mut Self::Elem;
}

/// Reference-counted tensor buffer for efficient memory sharing
pub struct SharedBuffer<T> {
    data: Arc<dyn TensorBuffer<Elem = T>>,
    offset: usize,
    len: usize,
}

impl<T: Clone + Send + Sync + 'static> SharedBuffer<T> {
    /// Create a new shared buffer from a tensor buffer
    pub fn new(buffer: Box<dyn TensorBuffer<Elem = T>>) -> Self {
        let len = buffer.len();
        Self {
            data: Arc::from(buffer),
            offset: 0,
            len,
        }
    }

    /// Create a view into this buffer (zero-copy)
    pub fn view(&self, offset: usize, len: usize) -> Result<Self> {
        if offset + len > self.len {
            return Err(TensorError::invalid_argument(format!(
                "View out of bounds: offset={offset}, len={len}, buffer_len={}",
                self.len
            )));
        }

        Ok(Self {
            data: Arc::clone(&self.data),
            offset: self.offset + offset,
            len,
        })
    }

    /// Get the reference count
    pub fn ref_count(&self) -> usize {
        Arc::strong_count(&self.data)
    }
}

/// CPU buffer implementation using Vec
pub struct CpuBuffer<T> {
    data: Vec<T>,
    device: Device,
}

impl<T: Clone + Send + Sync> CpuBuffer<T> {
    pub fn new(data: Vec<T>) -> Self {
        Self {
            data,
            device: Device::Cpu,
        }
    }

    pub fn zeros(len: usize) -> Self
    where
        T: Default,
    {
        Self {
            data: vec![T::default(); len],
            device: Device::Cpu,
        }
    }
}

impl<T: Clone + Send + Sync + 'static> TensorBuffer for CpuBuffer<T> {
    type Elem = T;

    fn device(&self) -> &Device {
        &self.device
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn size_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<T>()
    }

    fn clone_buffer(&self) -> Result<Box<dyn TensorBuffer<Elem = Self::Elem>>> {
        Ok(Box::new(Self {
            data: self.data.clone(),
            device: self.device,
        }))
    }

    fn view(&self, offset: usize, len: usize) -> Result<Box<dyn TensorBuffer<Elem = Self::Elem>>> {
        if offset + len > self.data.len() {
            return Err(TensorError::invalid_argument(format!(
                "View out of bounds: offset={offset}, len={len}, buffer_len={}",
                self.data.len()
            )));
        }

        Ok(Box::new(Self {
            data: self.data[offset..offset + len].to_vec(),
            device: self.device,
        }))
    }

    fn to_cpu(&self) -> Result<Vec<Self::Elem>> {
        Ok(self.data.clone())
    }

    unsafe fn as_ptr(&self) -> *const Self::Elem {
        self.data.as_ptr()
    }

    unsafe fn as_mut_ptr(&mut self) -> *mut Self::Elem {
        self.data.as_mut_ptr()
    }
}

type PoolKey = (Device, usize);
type PoolValue = Vec<Box<dyn std::any::Any + Send>>;
type PoolMap = std::collections::HashMap<PoolKey, PoolValue>;

/// Memory pool for efficient buffer allocation and reuse
pub struct MemoryPool {
    pools: std::sync::Mutex<PoolMap>,
    max_pool_size: usize,
}

impl MemoryPool {
    pub fn new(max_pool_size: usize) -> Self {
        Self {
            pools: std::sync::Mutex::new(std::collections::HashMap::new()),
            max_pool_size,
        }
    }

    /// Allocate a buffer from the pool or create a new one
    pub fn allocate<
        T: Clone + Send + Sync + Default + bytemuck::Pod + bytemuck::Zeroable + 'static,
    >(
        &self,
        device: Device,
        len: usize,
    ) -> Box<dyn TensorBuffer<Elem = T>> {
        let key = (device, std::mem::size_of::<T>());
        let mut pools = self.pools.lock().unwrap();

        if let Some(pool) = pools.get_mut(&key) {
            // Try to find a suitable buffer in the pool
            for i in 0..pool.len() {
                if let Some(buffer) = pool[i].downcast_ref::<CpuBuffer<T>>() {
                    if buffer.len() >= len {
                        let recycled = pool.swap_remove(i);
                        if let Ok(mut buffer) = recycled.downcast::<CpuBuffer<T>>() {
                            // Resize if needed
                            buffer.data.resize(len, T::default());
                            return buffer;
                        }
                    }
                }
            }
        }

        // Allocate new buffer
        match device {
            Device::Cpu => Box::new(CpuBuffer::zeros(len)),
            #[cfg(feature = "gpu")]
            Device::Gpu(id) => {
                use crate::gpu::buffer::GpuBuffer;
                Box::new(GpuBuffer::<T>::zeros(len, id).unwrap())
            }
            #[cfg(feature = "rocm")]
            Device::Rocm(id) => {
                use crate::gpu::buffer::GpuBuffer;
                Box::new(GpuBuffer::<T>::zeros(len, id).unwrap())
            }
        }
    }

    /// Return a buffer to the pool for reuse
    pub fn deallocate<T: 'static>(&self, device: Device, buffer: Box<dyn std::any::Any + Send>) {
        let key = (device, std::mem::size_of::<T>());
        let mut pools = self.pools.lock().unwrap();

        let pool = pools.entry(key).or_default();
        if pool.len() < self.max_pool_size {
            pool.push(buffer);
        }
    }
}

// Global memory pool instance
lazy_static::lazy_static! {
    pub static ref MEMORY_POOL: MemoryPool = MemoryPool::new(100);
}
