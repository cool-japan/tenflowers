use crate::{Device, Result, TensorError};
use std::collections::HashMap;
use std::sync::Arc;

/// Device-specific memory allocator trait
pub trait DeviceAllocator: Send + Sync {
    /// Allocate memory on the device
    fn allocate(&self, size: usize) -> Result<*mut u8>;

    /// Deallocate memory on the device
    ///
    /// # Safety
    /// The caller must ensure that:
    /// - The pointer was allocated by this allocator
    /// - The size matches the original allocation
    /// - The pointer is not used after deallocation
    unsafe fn deallocate(&self, ptr: *mut u8, size: usize);

    /// Copy memory from host to device
    fn copy_from_host(&self, dst: *mut u8, src: &[u8]) -> Result<()>;

    /// Copy memory from device to host
    fn copy_to_host(&self, dst: &mut [u8], src: *const u8) -> Result<()>;

    /// Copy memory within device
    fn copy_device_to_device(&self, dst: *mut u8, src: *const u8, size: usize) -> Result<()>;
}

/// Device-specific compute context
pub trait DeviceContext: Send + Sync {
    /// Get the device
    fn device(&self) -> &Device;

    /// Get the allocator for this device
    fn allocator(&self) -> &dyn DeviceAllocator;

    /// Synchronize device operations
    fn synchronize(&self) -> Result<()>;

    /// Create a compute stream/queue
    fn create_stream(&self) -> Result<Box<dyn DeviceStream>>;

    /// Get device properties
    fn properties(&self) -> DeviceProperties;
}

/// Device stream for async operations
pub trait DeviceStream: Send + Sync {
    /// Enqueue a kernel launch
    fn launch_kernel(&self, kernel: &dyn DeviceKernel, args: KernelArgs) -> Result<()>;

    /// Synchronize the stream
    fn synchronize(&self) -> Result<()>;

    /// Check if operations are complete
    fn is_complete(&self) -> bool;
}

/// Device kernel interface
pub trait DeviceKernel: Send + Sync {
    /// Get kernel name
    fn name(&self) -> &str;

    /// Get required shared memory size
    fn shared_memory_size(&self) -> usize;

    /// Get optimal block size
    fn optimal_block_size(&self) -> usize;
}

/// Kernel launch arguments
pub struct KernelArgs {
    pub grid_size: [u32; 3],
    pub block_size: [u32; 3],
    pub shared_memory: usize,
    pub inputs: Vec<*const u8>,
    pub outputs: Vec<*mut u8>,
    pub params: HashMap<String, KernelParam>,
}

/// Kernel parameter types
#[derive(Clone)]
pub enum KernelParam {
    I32(i32),
    U32(u32),
    F32(f32),
    F64(f64),
    Ptr(*const u8),
}

/// Device properties
#[derive(Debug, Clone)]
pub struct DeviceProperties {
    pub name: String,
    pub total_memory: usize,
    pub compute_capability: (u32, u32),
    pub max_threads_per_block: u32,
    pub max_blocks: [u32; 3],
    pub shared_memory_per_block: usize,
    pub warp_size: u32,
}

/// CPU device context implementation
pub struct CpuContext {
    allocator: CpuAllocator,
}

impl CpuContext {
    pub fn new() -> Self {
        Self {
            allocator: CpuAllocator,
        }
    }
}

impl Default for CpuContext {
    fn default() -> Self {
        Self::new()
    }
}

impl DeviceContext for CpuContext {
    fn device(&self) -> &Device {
        &Device::Cpu
    }

    fn allocator(&self) -> &dyn DeviceAllocator {
        &self.allocator
    }

    fn synchronize(&self) -> Result<()> {
        // CPU operations are synchronous
        Ok(())
    }

    fn create_stream(&self) -> Result<Box<dyn DeviceStream>> {
        Ok(Box::new(CpuStream))
    }

    fn properties(&self) -> DeviceProperties {
        DeviceProperties {
            name: "CPU".to_string(),
            total_memory: sys_info::mem_info()
                .map(|info| info.total as usize * 1024)
                .unwrap_or(0),
            compute_capability: (1, 0),
            max_threads_per_block: 1,
            max_blocks: [1, 1, 1],
            shared_memory_per_block: 0,
            warp_size: 1,
        }
    }
}

/// CPU allocator
struct CpuAllocator;

impl DeviceAllocator for CpuAllocator {
    fn allocate(&self, size: usize) -> Result<*mut u8> {
        if size == 0 {
            return Ok(std::ptr::null_mut());
        }

        let layout = std::alloc::Layout::from_size_align(size, 64)
            .map_err(|e| TensorError::allocation_error_simple(e.to_string()))?;

        let ptr = unsafe { std::alloc::alloc(layout) };
        if ptr.is_null() {
            Err(TensorError::allocation_error_simple(format!(
                "Failed to allocate {size} bytes"
            )))
        } else {
            Ok(ptr)
        }
    }

    unsafe fn deallocate(&self, ptr: *mut u8, size: usize) {
        if !ptr.is_null() && size > 0 {
            let layout = std::alloc::Layout::from_size_align_unchecked(size, 64);
            std::alloc::dealloc(ptr, layout);
        }
    }

    fn copy_from_host(&self, dst: *mut u8, src: &[u8]) -> Result<()> {
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len());
        }
        Ok(())
    }

    fn copy_to_host(&self, dst: &mut [u8], src: *const u8) -> Result<()> {
        unsafe {
            std::ptr::copy_nonoverlapping(src, dst.as_mut_ptr(), dst.len());
        }
        Ok(())
    }

    fn copy_device_to_device(&self, dst: *mut u8, src: *const u8, size: usize) -> Result<()> {
        unsafe {
            std::ptr::copy_nonoverlapping(src, dst, size);
        }
        Ok(())
    }
}

/// CPU stream (synchronous)
struct CpuStream;

impl DeviceStream for CpuStream {
    fn launch_kernel(&self, _kernel: &dyn DeviceKernel, _args: KernelArgs) -> Result<()> {
        // CPU kernels are executed directly, not through stream
        Ok(())
    }

    fn synchronize(&self) -> Result<()> {
        Ok(())
    }

    fn is_complete(&self) -> bool {
        true
    }
}

/// Device manager for context creation and caching
pub struct DeviceManager {
    contexts: std::sync::RwLock<HashMap<Device, Arc<dyn DeviceContext>>>,
}

impl DeviceManager {
    pub fn new() -> Self {
        Self {
            contexts: std::sync::RwLock::new(HashMap::new()),
        }
    }
}

impl Default for DeviceManager {
    fn default() -> Self {
        Self::new()
    }
}

impl DeviceManager {
    /// Get or create a device context
    pub fn get_context(&self, device: &Device) -> Result<Arc<dyn DeviceContext>> {
        // Check cache first
        {
            let contexts = self.contexts.read().unwrap();
            if let Some(ctx) = contexts.get(device) {
                return Ok(Arc::clone(ctx));
            }
        }

        // Create new context
        let context: Arc<dyn DeviceContext> = match device {
            Device::Cpu => Arc::new(CpuContext::new()),
            #[cfg(feature = "gpu")]
            Device::Gpu(id) => Arc::new(GpuContext::new(*id)?),
            #[cfg(feature = "rocm")]
            Device::Rocm(id) => Arc::new(GpuContext::new(*id)?), // Use same GPU context for ROCm
        };

        // Cache it
        {
            let mut contexts = self.contexts.write().unwrap();
            contexts.insert(*device, Arc::clone(&context));
        }

        Ok(context)
    }
}

// Global device manager
lazy_static::lazy_static! {
    pub static ref DEVICE_MANAGER: DeviceManager = DeviceManager::new();
}

/// GPU device context implementation
#[cfg(feature = "gpu")]
pub struct GpuContext {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    adapter: wgpu::Adapter,
    allocator: GpuAllocator,
    device_enum: Device,
}

#[cfg(feature = "gpu")]
impl GpuContext {
    pub fn new(device_id: usize) -> Result<Self> {
        pollster::block_on(Self::new_async(device_id))
    }

    async fn new_async(device_id: usize) -> Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| {
                TensorError::device_error_simple("Failed to find suitable GPU adapter".to_string())
            })?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some(&format!("TenfloweRS GPU Device {device_id}")),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|e| {
                TensorError::device_error_simple(format!("Failed to create GPU device: {e}"))
            })?;

        let device = Arc::new(device);
        let queue = Arc::new(queue);
        let allocator = GpuAllocator::new(Arc::clone(&device), Arc::clone(&queue));

        Ok(Self {
            device,
            queue,
            adapter,
            allocator,
            device_enum: Device::Gpu(device_id),
        })
    }
}

#[cfg(feature = "gpu")]
impl DeviceContext for GpuContext {
    fn device(&self) -> &Device {
        &self.device_enum
    }

    fn allocator(&self) -> &dyn DeviceAllocator {
        &self.allocator
    }

    fn synchronize(&self) -> Result<()> {
        self.device.poll(wgpu::Maintain::Wait);
        Ok(())
    }

    fn create_stream(&self) -> Result<Box<dyn DeviceStream>> {
        Ok(Box::new(GpuStream::new(
            Arc::clone(&self.device),
            Arc::clone(&self.queue),
        )))
    }

    fn properties(&self) -> DeviceProperties {
        let info = self.adapter.get_info();
        let limits = self.adapter.limits();

        DeviceProperties {
            name: info.name,
            total_memory: 0,            // WGPU doesn't expose memory info
            compute_capability: (1, 0), // Generic capability
            max_threads_per_block: limits.max_compute_workgroup_size_x,
            max_blocks: [
                limits.max_compute_workgroups_per_dimension,
                limits.max_compute_workgroups_per_dimension,
                limits.max_compute_workgroups_per_dimension,
            ],
            shared_memory_per_block: limits.max_compute_workgroup_storage_size as usize,
            warp_size: 32, // Common warp size
        }
    }
}

/// GPU allocator implementation
#[cfg(feature = "gpu")]
struct GpuAllocator {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
}

#[cfg(feature = "gpu")]
impl GpuAllocator {
    fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        Self { device, queue }
    }
}

#[cfg(feature = "gpu")]
impl DeviceAllocator for GpuAllocator {
    fn allocate(&self, size: usize) -> Result<*mut u8> {
        // WGPU doesn't expose raw pointers, return a placeholder
        // Real GPU allocation happens through wgpu::Buffer
        Ok(size as *mut u8) // Placeholder pointer
    }

    unsafe fn deallocate(&self, _ptr: *mut u8, _size: usize) {
        // GPU buffers are managed by WGPU automatically
    }

    fn copy_from_host(&self, _dst: *mut u8, _src: &[u8]) -> Result<()> {
        // GPU memory operations are handled by buffer operations
        Ok(())
    }

    fn copy_to_host(&self, _dst: &mut [u8], _src: *const u8) -> Result<()> {
        // GPU memory operations are handled by buffer operations
        Ok(())
    }

    fn copy_device_to_device(&self, _dst: *mut u8, _src: *const u8, _size: usize) -> Result<()> {
        // GPU memory operations are handled by buffer operations
        Ok(())
    }
}

/// GPU stream implementation
#[cfg(feature = "gpu")]
struct GpuStream {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
}

#[cfg(feature = "gpu")]
impl GpuStream {
    fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        Self { device, queue }
    }
}

#[cfg(feature = "gpu")]
impl DeviceStream for GpuStream {
    fn launch_kernel(&self, _kernel: &dyn DeviceKernel, _args: KernelArgs) -> Result<()> {
        // Kernel execution will be implemented with compute shaders
        Ok(())
    }

    fn synchronize(&self) -> Result<()> {
        self.device.poll(wgpu::Maintain::Wait);
        Ok(())
    }

    fn is_complete(&self) -> bool {
        // For WGPU, operations are generally synchronous
        true
    }
}

/// Helper function to get GPU context with backend selection
#[cfg(feature = "gpu")]
pub fn get_gpu_context(device_id: usize) -> Result<GpuContextInfo> {
    let device = Device::Gpu(device_id);
    let ctx = DEVICE_MANAGER.get_context(&device)?;

    // Create a new GPU context if needed (simple approach)
    let gpu_ctx = GpuContext::new(device_id)?;
    Ok(GpuContextInfo {
        device: gpu_ctx.device.clone(),
        queue: gpu_ctx.queue.clone(),
    })
}

/// Enhanced GPU context with backend selection capabilities
#[cfg(any(feature = "gpu", feature = "cudnn"))]
pub fn get_enhanced_gpu_context(device_id: usize) -> Result<EnhancedGpuContext> {
    // Check if cuDNN is available and requested
    #[cfg(all(feature = "cudnn", any(target_os = "linux", target_os = "windows")))]
    {
        if crate::gpu::cudnn::CudnnContext::is_available() {
            let mut cudnn_ctx = crate::gpu::cudnn::global_cudnn_context();
            let cudnn_handle = cudnn_ctx.get_handle(device_id)?;

            // Also get WGPU context for fallback
            let wgpu_ctx = get_gpu_context(device_id)?;

            return Ok(EnhancedGpuContext {
                device_id,
                backend: GpuBackend::CuDNN(cudnn_handle),
                wgpu_fallback: Some(wgpu_ctx),
            });
        }
    }

    // Fall back to WGPU
    #[cfg(feature = "gpu")]
    {
        let wgpu_ctx = get_gpu_context(device_id)?;
        Ok(EnhancedGpuContext {
            device_id,
            backend: GpuBackend::WGPU(wgpu_ctx),
            wgpu_fallback: None,
        })
    }
    #[cfg(not(feature = "gpu"))]
    {
        Err(TensorError::unsupported_operation_simple(
            "No GPU backend available",
        ))
    }
}

#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
pub struct GpuContextInfo {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
}

/// GPU backend selection enum
#[cfg(any(feature = "gpu", feature = "cudnn"))]
#[derive(Debug, Clone)]
pub enum GpuBackend {
    #[cfg(feature = "gpu")]
    WGPU(GpuContextInfo),
    #[cfg(all(feature = "cudnn", any(target_os = "linux", target_os = "windows")))]
    CuDNN(Arc<crate::gpu::cudnn::CudnnHandle>),
}

/// Enhanced GPU context with backend selection capabilities
#[cfg(any(feature = "gpu", feature = "cudnn"))]
#[derive(Debug, Clone)]
pub struct EnhancedGpuContext {
    pub device_id: usize,
    pub backend: GpuBackend,
    pub wgpu_fallback: Option<GpuContextInfo>,
}

#[cfg(any(feature = "gpu", feature = "cudnn"))]
impl EnhancedGpuContext {
    /// Check if cuDNN backend is being used
    #[cfg(all(feature = "cudnn", any(target_os = "linux", target_os = "windows")))]
    pub fn is_cudnn(&self) -> bool {
        matches!(self.backend, GpuBackend::CuDNN(_))
    }

    /// Check if WGPU backend is being used
    #[cfg(feature = "gpu")]
    pub fn is_wgpu(&self) -> bool {
        matches!(self.backend, GpuBackend::WGPU(_))
    }

    /// Get cuDNN handle if available
    #[cfg(all(feature = "cudnn", any(target_os = "linux", target_os = "windows")))]
    pub fn get_cudnn_handle(&self) -> Option<&Arc<crate::gpu::cudnn::CudnnHandle>> {
        match &self.backend {
            GpuBackend::CuDNN(handle) => Some(handle),
            _ => None,
        }
    }

    /// Get WGPU context if available
    #[cfg(feature = "gpu")]
    pub fn get_wgpu_context(&self) -> Option<&GpuContextInfo> {
        match &self.backend {
            GpuBackend::WGPU(ctx) => Some(ctx),
            #[cfg(all(feature = "cudnn", any(target_os = "linux", target_os = "windows")))]
            _ => self.wgpu_fallback.as_ref(),
        }
    }

    /// Get device ID
    pub fn device_id(&self) -> usize {
        self.device_id
    }
}

// Add sys-info dependency detection
#[cfg(not(target_arch = "wasm32"))]
extern crate sys_info;

#[cfg(target_arch = "wasm32")]
mod sys_info {
    pub struct MemInfo {
        pub total: u64,
    }

    pub fn mem_info() -> Result<MemInfo, &'static str> {
        Ok(MemInfo {
            total: 1024 * 1024 * 1024,
        }) // 1GB default for WASM
    }
}
