use crate::{buffer::TensorBuffer, Device, Result, TensorError};
use scirs2_autograd::ndarray::ArrayD;
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// A GPU buffer that stores tensor data on the GPU device
#[cfg(feature = "gpu")]
#[derive(Debug)]
pub struct GpuBuffer<T> {
    buffer: Arc<wgpu::Buffer>,
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    device_enum: Device,
    len: usize,
    is_pinned: bool,
    _phantom: std::marker::PhantomData<T>,
}

/// A view into a GPU buffer that references a portion of the original buffer
pub struct GpuBufferView<T> {
    parent_buffer: Arc<GpuBuffer<T>>,
    offset: usize,
    len: usize,
    device_enum: Device,
    _phantom: std::marker::PhantomData<T>,
}

#[cfg(feature = "gpu")]
impl<T: bytemuck::Pod + bytemuck::Zeroable + Clone + Send + Sync + 'static> TensorBuffer
    for GpuBufferView<T>
{
    type Elem = T;

    fn device(&self) -> &Device {
        &self.device_enum
    }

    fn len(&self) -> usize {
        self.len
    }

    fn size_bytes(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }

    fn view(&self, offset: usize, len: usize) -> Result<Box<dyn TensorBuffer<Elem = T>>> {
        if offset + len > self.len {
            return Err(TensorError::invalid_operation_simple(format!(
                "View out of bounds: {}+{} > {}",
                offset, len, self.len
            )));
        }
        Ok(Box::new(GpuBufferView::new(
            Arc::clone(&self.parent_buffer),
            self.offset + offset,
            len,
        )?))
    }

    fn to_cpu(&self) -> Result<Vec<T>> {
        self.parent_buffer.to_cpu()
    }

    unsafe fn as_ptr(&self) -> *const T {
        // GPU buffers don't have CPU-accessible pointers
        std::ptr::null()
    }

    unsafe fn as_mut_ptr(&mut self) -> *mut T {
        // GPU buffers don't have CPU-accessible pointers
        std::ptr::null_mut()
    }

    fn clone_buffer(&self) -> Result<Box<dyn TensorBuffer<Elem = T>>> {
        Ok(Box::new(self.clone()))
    }
}

impl<T: bytemuck::Pod + bytemuck::Zeroable + Clone + Send + Sync + 'static> GpuBufferView<T> {
    pub fn new(parent: Arc<GpuBuffer<T>>, offset: usize, len: usize) -> Result<Self> {
        if offset + len > parent.len() {
            return Err(TensorError::invalid_operation_simple(format!(
                "Buffer view out of bounds: {}+{} > {}",
                offset,
                len,
                parent.len()
            )));
        }
        Ok(Self {
            device_enum: parent.device_enum.clone(),
            parent_buffer: parent,
            offset,
            len,
            _phantom: std::marker::PhantomData,
        })
    }

    #[inline]
    pub fn parent(&self) -> &GpuBuffer<T> {
        &self.parent_buffer
    }

    #[inline]
    pub fn offset(&self) -> usize {
        self.offset
    }

    #[inline]
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.parent_buffer.buffer
    }

    #[inline]
    pub fn device(&self) -> &wgpu::Device {
        &self.parent_buffer.device
    }

    #[inline]
    pub fn queue(&self) -> &wgpu::Queue {
        &self.parent_buffer.queue
    }

    pub fn to_cpu_array(&self) -> Result<ArrayD<T>> {
        // For now, we need to copy the entire parent buffer and then slice
        // In the future, we could optimize this to only copy the view portion
        let full_data = self.parent_buffer.to_cpu()?;
        let view_data = full_data[self.offset..self.offset + self.len].to_vec();
        Ok(ndarray::Array1::from(view_data).into_dyn())
    }
}

impl<T> Clone for GpuBufferView<T> {
    fn clone(&self) -> Self {
        Self {
            parent_buffer: Arc::clone(&self.parent_buffer),
            offset: self.offset,
            len: self.len,
            device_enum: self.device_enum.clone(),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T> Clone for GpuBuffer<T> {
    fn clone(&self) -> Self {
        Self {
            buffer: Arc::clone(&self.buffer),
            device: Arc::clone(&self.device),
            queue: Arc::clone(&self.queue),
            device_enum: self.device_enum.clone(),
            len: self.len,
            is_pinned: self.is_pinned,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: bytemuck::Pod + bytemuck::Zeroable + Clone + Send + Sync + 'static> GpuBuffer<T> {
    /// Create a new GPU buffer filled with zeros
    pub fn zeros(len: usize, device_id: usize) -> Result<Self> {
        use wgpu::util::DeviceExt;

        // Get the global GPU context
        let context = crate::gpu::GpuContext::global()?;
        let device = &context.device;
        let queue = &context.queue;

        // Create a buffer filled with zeros
        let zeros_data = vec![T::zeroed(); len];
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gpu_buffer_zeros"),
            contents: bytemuck::cast_slice(&zeros_data),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        Ok(Self {
            buffer: Arc::new(buffer),
            device: Arc::clone(&context.device),
            queue: Arc::clone(&context.queue),
            device_enum: Device::Gpu(device_id),
            len,
            is_pinned: false,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Create a GpuBuffer from an existing wgpu::Buffer
    pub fn from_wgpu_buffer(
        buffer: wgpu::Buffer,
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        device_enum: Device,
        len: usize,
    ) -> Self {
        Self {
            buffer: Arc::new(buffer),
            device,
            queue,
            device_enum,
            len,
            is_pinned: false,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create a GpuBuffer from an existing raw wgpu::Buffer (alias for from_wgpu_buffer)
    pub fn from_raw_buffer(
        buffer: wgpu::Buffer,
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        device_enum: Device,
        len: usize,
    ) -> Self {
        Self::from_wgpu_buffer(buffer, device, queue, device_enum, len)
    }

    /// Create a GpuBuffer from an Arc<wgpu::Buffer>
    pub fn from_shared_buffer(
        buffer: Arc<wgpu::Buffer>,
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        device_enum: Device,
        len: usize,
    ) -> Self {
        Self {
            buffer,
            device,
            queue,
            device_enum,
            len,
            is_pinned: false,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn from_cpu_array(array: &ArrayD<T>, device_id: usize) -> Result<Self> {
        use wgpu::util::DeviceExt;

        // Get the global GPU context
        let context = crate::gpu::GpuContext::global()?;
        let device = &context.device;
        let queue = &context.queue;

        // Convert array to slice - handle both contiguous and non-contiguous arrays
        let slice = if array.is_standard_layout() {
            array.as_slice().ok_or_else(|| {
                TensorError::invalid_operation_simple(
                    "Cannot convert non-contiguous array to slice".to_string(),
                )
            })?
        } else {
            // For non-contiguous arrays, we need to collect into a vector first
            let data: Vec<T> = array.iter().cloned().collect();
            return Self::from_slice(&data, &Device::Gpu(device_id));
        };

        // Create GPU buffer from slice
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gpu_buffer_from_cpu_array"),
            contents: bytemuck::cast_slice(slice),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        Ok(Self {
            buffer: Arc::new(buffer),
            device: Arc::clone(&context.device),
            queue: Arc::clone(&context.queue),
            device_enum: Device::Gpu(device_id),
            len: array.len(),
            is_pinned: false,
            _phantom: std::marker::PhantomData,
        })
    }

    pub fn to_cpu_array(&self) -> Result<ArrayD<T>> {
        let data = self.to_cpu()?;
        Ok(ndarray::Array1::from(data).into_dyn())
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn size_bytes(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }

    pub fn from_slice(slice: &[T], device: &Device) -> Result<Self> {
        match device {
            Device::Gpu(_device_id) => {
                // Get the global GPU context
                let context = crate::gpu::GpuContext::global()?;
                let gpu_device = &context.device;
                let queue = &context.queue;

                let buffer = gpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("tensor_buffer_from_slice"),
                    contents: bytemuck::cast_slice(slice),
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                });

                Ok(Self {
                    buffer: Arc::new(buffer),
                    device: Arc::clone(&context.device),
                    queue: Arc::clone(&context.queue),
                    device_enum: device.clone(),
                    len: slice.len(),
                    is_pinned: false,
                    _phantom: std::marker::PhantomData,
                })
            }
            _ => Err(TensorError::invalid_operation_simple(
                "Expected GPU device".to_string(),
            )),
        }
    }

    pub fn transfer_to_device(&self, target_device: &Device) -> Result<Self> {
        match target_device {
            Device::Gpu(_device_id) => {
                // Transfer data via CPU for now - could be optimized for peer-to-peer transfer
                let cpu_data = self.to_cpu()?;
                Self::from_slice(&cpu_data, target_device)
            }
            _ => Err(TensorError::invalid_operation_simple(
                "Expected GPU device".to_string(),
            )),
        }
    }

    pub fn to_cpu(&self) -> Result<Vec<T>> {
        // Create a staging buffer for reading data back
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_buffer"),
            size: self.size_bytes() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Copy from GPU buffer to staging buffer
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gpu_to_cpu_encoder"),
            });
        encoder.copy_buffer_to_buffer(
            &*self.buffer,
            0,
            &staging_buffer,
            0,
            self.size_bytes() as u64,
        );
        self.queue.submit(Some(encoder.finish()));

        // Map and read the staging buffer
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        self.device.poll(wgpu::Maintain::Wait);

        match futures::executor::block_on(receiver) {
            Ok(Ok(())) => {
                let data = buffer_slice.get_mapped_range();
                let result = bytemuck::cast_slice(&data).to_vec();
                drop(data);
                staging_buffer.unmap();
                Ok(result)
            }
            _ => Err(TensorError::invalid_operation_simple(
                "Failed to read GPU buffer".to_string(),
            )),
        }
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        &*self.buffer
    }

    pub fn buffer_arc(&self) -> Arc<wgpu::Buffer> {
        Arc::clone(&self.buffer)
    }

    pub fn device_enum(&self) -> Device {
        self.device_enum.clone()
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    #[inline]
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    #[inline]
    pub fn is_pinned(&self) -> bool {
        self.is_pinned
    }

    pub fn from_cpu_array_pinned(array: &ArrayD<T>, device_id: usize) -> Result<Self> {
        let mut buffer = Self::from_cpu_array(array, device_id)?;
        buffer.is_pinned = true;
        Ok(buffer)
    }

    pub fn zeros_pinned(len: usize, device_id: usize) -> Result<Self> {
        let mut buffer = Self::zeros(len, device_id)?;
        buffer.is_pinned = true;
        Ok(buffer)
    }
}

#[cfg(feature = "gpu")]
impl<T: bytemuck::Pod + bytemuck::Zeroable + Clone + Send + Sync + 'static> TensorBuffer
    for GpuBuffer<T>
{
    type Elem = T;

    fn device(&self) -> &Device {
        &self.device_enum
    }

    fn len(&self) -> usize {
        self.len
    }

    fn size_bytes(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }

    fn view(&self, offset: usize, len: usize) -> Result<Box<dyn TensorBuffer<Elem = T>>> {
        if offset + len > self.len {
            return Err(TensorError::invalid_operation_simple(format!(
                "View out of bounds: {}+{} > {}",
                offset, len, self.len
            )));
        }
        Ok(Box::new(GpuBufferView::new(
            Arc::new(self.clone()),
            offset,
            len,
        )?))
    }

    fn to_cpu(&self) -> Result<Vec<T>> {
        // Call the actual GpuBuffer::to_cpu method
        GpuBuffer::to_cpu(self)
    }

    unsafe fn as_ptr(&self) -> *const T {
        // GPU buffers don't have CPU-accessible pointers
        std::ptr::null()
    }

    unsafe fn as_mut_ptr(&mut self) -> *mut T {
        // GPU buffers don't have CPU-accessible pointers
        std::ptr::null_mut()
    }

    fn clone_buffer(&self) -> Result<Box<dyn TensorBuffer<Elem = T>>> {
        Ok(Box::new(self.clone()))
    }
}
/// Trait for GPU buffer operations
pub trait GpuBufferOps<T> {
    fn create_buffer(&self, size: usize) -> Result<GpuBuffer<T>>;
    fn copy_from_host(&self, data: &[T]) -> Result<GpuBuffer<T>>;
    fn copy_to_host(&self, buffer: &GpuBuffer<T>) -> Result<Vec<T>>;
}

/// Manager for GPU buffer allocation and memory management
pub struct BufferManager {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
}

impl BufferManager {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        Self { device, queue }
    }

    pub fn allocate<T>(&self, size: usize) -> Result<GpuBuffer<T>>
    where
        T: bytemuck::Pod + bytemuck::Zeroable + Clone + Send + Sync + 'static,
    {
        GpuBuffer::zeros(size, 0)
    }
}
