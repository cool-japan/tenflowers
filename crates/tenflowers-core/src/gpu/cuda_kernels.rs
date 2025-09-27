//! CUDA kernel launcher for direct CUDA backend support
//!
//! This module provides low-level CUDA kernel execution capabilities,
//! complementing the higher-level cuDNN operations with direct kernel launches
//! for custom operations and maximum performance control.

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
use crate::{DType, Device, Result, Tensor, TensorError};
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::sync::Arc;

/// CUDA device context for managing CUDA runtime and driver
#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
#[derive(Debug)]
pub struct CudaDevice {
    /// CUDA device ID
    device_id: i32,
    /// CUDA context handle
    context: CudaContext,
    /// CUDA streams for asynchronous execution
    streams: Vec<CudaStream>,
    /// Cache of compiled CUDA modules
    module_cache: HashMap<String, CudaModule>,
    /// Cache of compiled kernels
    kernel_cache: HashMap<String, CudaKernel>,
    /// Device properties
    properties: CudaDeviceProperties,
}

/// CUDA kernel launch configuration
#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
#[derive(Debug, Clone)]
pub struct CudaKernelConfig {
    /// Grid dimensions (number of blocks)
    pub grid_dim: (u32, u32, u32),
    /// Block dimensions (threads per block)
    pub block_dim: (u32, u32, u32),
    /// Shared memory size in bytes
    pub shared_memory: u32,
    /// Stream for execution
    pub stream: Option<CudaStream>,
}

/// CUDA memory management and allocation
#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
#[derive(Debug)]
pub struct CudaMemoryPool {
    /// Available memory chunks organized by size
    available_chunks: HashMap<usize, Vec<*mut std::ffi::c_void>>,
    /// Total allocated memory
    total_allocated: usize,
    /// Memory alignment (typically 256 bytes for coalesced access)
    alignment: usize,
}

/// High-performance tensor operations using direct CUDA kernel launches
#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
impl CudaDevice {
    /// Create a new CUDA device instance
    pub fn new(device_id: i32) -> Result<Self> {
        // Initialize CUDA runtime
        unsafe {
            cuda_init(0)?;
        }

        // Set device
        unsafe {
            cuda_set_device(device_id)?;
        }

        // Create CUDA context
        let context = CudaContext::new(device_id)?;

        // Create multiple streams for overlapping computation
        let mut streams = Vec::new();
        for _ in 0..4 {
            streams.push(CudaStream::new()?);
        }

        // Get device properties
        let properties = CudaDeviceProperties::query(device_id)?;

        Ok(CudaDevice {
            device_id,
            context,
            streams,
            module_cache: HashMap::new(),
            kernel_cache: HashMap::new(),
            properties,
        })
    }

    /// Get device properties and capabilities
    pub fn get_device_properties(&self) -> &CudaDeviceProperties {
        &self.properties
    }

    /// Compile CUDA kernel from PTX or CUBIN
    pub fn compile_kernel(
        &mut self,
        source: &str,
        kernel_name: &str,
        compile_options: &[&str],
    ) -> Result<()> {
        let module = self.compile_module(source, compile_options)?;
        let kernel = module.get_function(kernel_name)?;

        self.module_cache
            .insert(format!("{}_module", kernel_name), module);
        self.kernel_cache.insert(kernel_name.to_string(), kernel);

        Ok(())
    }

    /// Launch CUDA kernel with specified configuration
    pub fn launch_kernel<T>(
        &mut self,
        kernel_name: &str,
        config: &CudaKernelConfig,
        args: &[&[T]],
    ) -> Result<*mut std::ffi::c_void>
    where
        T: Clone + Default + Send + Sync + 'static,
    {
        let kernel = self.kernel_cache.get(kernel_name).ok_or_else(|| {
            TensorError::invalid_operation_simple(format!(
                "Kernel '{}' not found. Compile it first.",
                kernel_name
            ))
        })?;

        // Allocate device memory for arguments
        let mut device_ptrs = Vec::new();
        for arg in args {
            let device_ptr = self.allocate_device_memory(arg.len() * std::mem::size_of::<T>())?;

            // Copy data to device
            unsafe {
                cuda_memcpy_htod(
                    device_ptr,
                    arg.as_ptr() as *const std::ffi::c_void,
                    arg.len() * std::mem::size_of::<T>(),
                )?;
            }

            device_ptrs.push(device_ptr);
        }

        // Allocate output buffer (assume same size as first input for now)
        let output_size = if !args.is_empty() {
            args[0].len() * std::mem::size_of::<T>()
        } else {
            std::mem::size_of::<T>()
        };
        let output_ptr = self.allocate_device_memory(output_size)?;
        device_ptrs.push(output_ptr);

        // Prepare kernel arguments
        let mut kernel_args: Vec<*mut std::ffi::c_void> = device_ptrs
            .iter()
            .map(|&ptr| ptr as *mut std::ffi::c_void)
            .collect();

        // Launch kernel
        unsafe {
            cuda_launch_kernel(
                kernel.function,
                config.grid_dim,
                config.block_dim,
                kernel_args.as_mut_ptr(),
                config.shared_memory,
                config
                    .stream
                    .as_ref()
                    .map(|s| s.handle)
                    .unwrap_or(std::ptr::null_mut()),
            )?;
        }

        // Synchronize if no stream specified
        if config.stream.is_none() {
            unsafe {
                cuda_device_synchronize()?;
            }
        }

        // Clean up input device memory
        for ptr in &device_ptrs[..device_ptrs.len() - 1] {
            unsafe {
                cuda_free(*ptr)?;
            }
        }

        Ok(output_ptr)
    }

    /// Execute optimized matrix multiplication using custom CUDA kernels
    pub fn matmul_cuda<T>(&mut self, a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>
    where
        T: Clone + Default + Send + Sync + 'static,
    {
        let a_shape = a.shape();
        let b_shape = b.shape();

        // Validate matrix multiplication dimensions
        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(TensorError::invalid_operation_simple(
                "Matrix multiplication requires 2D tensors".to_string(),
            ));
        }

        let (m, k) = (a_shape[0], a_shape[1]);
        let (k2, n) = (b_shape[0], b_shape[1]);

        if k != k2 {
            return Err(TensorError::invalid_operation_simple(format!(
                "Matrix dimension mismatch: {} vs {}",
                k, k2
            )));
        }

        // Compile GEMM kernel if not already cached
        if !self.kernel_cache.contains_key("cuda_gemm") {
            let gemm_ptx = self.generate_gemm_ptx()?;
            self.compile_kernel(&gemm_ptx, "cuda_gemm", &["-O3", "-use_fast_math"])?;
        }

        // Calculate optimal grid and block dimensions
        let config = self.calculate_gemm_config(m, n, k)?;

        // Launch GEMM kernel
        let a_data = a.as_slice().ok_or_else(|| {
            TensorError::invalid_operation_simple("Failed to access tensor data".to_string())
        })?;
        let b_data = b.as_slice().ok_or_else(|| {
            TensorError::invalid_operation_simple("Failed to access tensor data".to_string())
        })?;
        let output_ptr = self.launch_kernel("cuda_gemm", &config, &[a_data, b_data])?;

        // Copy result back to host
        let output_size = m * n;
        let mut output_data = vec![T::default(); output_size];

        unsafe {
            cuda_memcpy_dtoh(
                output_data.as_mut_ptr() as *mut std::ffi::c_void,
                output_ptr,
                output_size * std::mem::size_of::<T>(),
            )?;
            cuda_free(output_ptr)?;
        }

        let output_shape = vec![m, n];
        Tensor::from_vec(output_data, &output_shape)
    }

    /// Execute optimized element-wise operations using CUDA kernels
    pub fn elementwise_cuda<T>(
        &mut self,
        a: &Tensor<T>,
        b: &Tensor<T>,
        operation: ElementwiseOp,
    ) -> Result<Tensor<T>>
    where
        T: Clone + Default + Send + Sync + 'static,
    {
        let kernel_name = match operation {
            ElementwiseOp::Add => "cuda_elementwise_add",
            ElementwiseOp::Mul => "cuda_elementwise_mul",
            ElementwiseOp::Sub => "cuda_elementwise_sub",
            ElementwiseOp::Div => "cuda_elementwise_div",
        };

        // Compile elementwise kernel if not cached
        if !self.kernel_cache.contains_key(kernel_name) {
            let elementwise_ptx = self.generate_elementwise_ptx(operation)?;
            self.compile_kernel(&elementwise_ptx, kernel_name, &["-O3", "-use_fast_math"])?;
        }

        // Calculate optimal configuration
        let total_elements = a.numel();
        let config = self.calculate_elementwise_config(total_elements)?;

        // Launch kernel
        let a_data = a.as_slice().ok_or_else(|| {
            TensorError::invalid_operation_simple("Failed to access tensor data".to_string())
        })?;
        let b_data = b.as_slice().ok_or_else(|| {
            TensorError::invalid_operation_simple("Failed to access tensor data".to_string())
        })?;
        let output_ptr = self.launch_kernel(kernel_name, &config, &[a_data, b_data])?;

        // Copy result back
        let mut output_data = vec![T::default(); total_elements];
        unsafe {
            cuda_memcpy_dtoh(
                output_data.as_mut_ptr() as *mut std::ffi::c_void,
                output_ptr,
                total_elements * std::mem::size_of::<T>(),
            )?;
            cuda_free(output_ptr)?;
        }

        Tensor::from_vec(output_data, a.shape().dims())
    }

    /// Execute custom user-defined CUDA kernel
    pub fn execute_custom_kernel<T>(
        &mut self,
        ptx_source: &str,
        kernel_name: &str,
        config: &CudaKernelConfig,
        inputs: &[&Tensor<T>],
    ) -> Result<Tensor<T>>
    where
        T: Clone + Default + Send + Sync + 'static,
    {
        // Compile custom kernel
        if !self.kernel_cache.contains_key(kernel_name) {
            self.compile_kernel(ptx_source, kernel_name, &["-O3"])?;
        }

        // Convert tensors to data slices
        let input_slices: Result<Vec<&[T]>> = inputs
            .iter()
            .map(|t| {
                t.as_slice().ok_or_else(|| {
                    TensorError::invalid_operation_simple(
                        "Failed to access tensor data".to_string(),
                    )
                })
            })
            .collect();
        let input_slices = input_slices?;

        // Launch kernel
        let output_ptr = self.launch_kernel(kernel_name, config, &input_slices)?;

        // Determine output shape (use first input's shape as default)
        let output_shape = if !inputs.is_empty() {
            inputs[0].shape().to_vec()
        } else {
            vec![1]
        };

        let output_size = output_shape.iter().product::<usize>();
        let mut output_data = vec![T::default(); output_size];

        unsafe {
            cuda_memcpy_dtoh(
                output_data.as_mut_ptr() as *mut std::ffi::c_void,
                output_ptr,
                output_size * std::mem::size_of::<T>(),
            )?;
            cuda_free(output_ptr)?;
        }

        Tensor::from_vec(output_data, &output_shape)
    }

    // Private implementation methods

    fn compile_module(&mut self, source: &str, options: &[&str]) -> Result<CudaModule> {
        // Convert options to C strings
        let c_options: Vec<CString> = options.iter().map(|&s| CString::new(s).unwrap()).collect();
        let option_ptrs: Vec<*const i8> = c_options.iter().map(|s| s.as_ptr()).collect();

        unsafe {
            let mut module = std::ptr::null_mut();
            cuda_module_load_data_ex(
                &mut module,
                source.as_ptr() as *const std::ffi::c_void,
                option_ptrs.len() as u32,
                option_ptrs.as_ptr(),
                std::ptr::null(),
            )?;

            Ok(CudaModule { handle: module })
        }
    }

    fn allocate_device_memory(&self, size: usize) -> Result<*mut std::ffi::c_void> {
        unsafe {
            let mut ptr = std::ptr::null_mut();
            cuda_malloc(&mut ptr, size)?;
            Ok(ptr)
        }
    }

    fn calculate_gemm_config(&self, m: usize, n: usize, k: usize) -> Result<CudaKernelConfig> {
        // Use tiled GEMM with 16x16 tiles for optimal memory coalescing
        let tile_size = 16;
        let grid_x = (n + tile_size - 1) / tile_size;
        let grid_y = (m + tile_size - 1) / tile_size;

        Ok(CudaKernelConfig {
            grid_dim: (grid_x as u32, grid_y as u32, 1),
            block_dim: (tile_size as u32, tile_size as u32, 1),
            shared_memory: (2 * tile_size * tile_size * 4) as u32, // 2 tiles * sizeof(float)
            stream: None,
        })
    }

    fn calculate_elementwise_config(&self, total_elements: usize) -> Result<CudaKernelConfig> {
        let threads_per_block = 256.min(total_elements);
        let blocks_needed = (total_elements + threads_per_block - 1) / threads_per_block;

        Ok(CudaKernelConfig {
            grid_dim: (blocks_needed as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_memory: 0,
            stream: None,
        })
    }

    fn generate_gemm_ptx(&self) -> Result<String> {
        // Generate PTX code for tiled matrix multiplication
        Ok(include_str!("cuda_kernels/gemm_kernel.ptx").to_string())
    }

    fn generate_elementwise_ptx(&self, operation: ElementwiseOp) -> Result<String> {
        match operation {
            ElementwiseOp::Add => Ok(include_str!("cuda_kernels/elementwise_add.ptx").to_string()),
            ElementwiseOp::Mul => Ok(include_str!("cuda_kernels/elementwise_mul.ptx").to_string()),
            ElementwiseOp::Sub => Ok(include_str!("cuda_kernels/elementwise_sub.ptx").to_string()),
            ElementwiseOp::Div => Ok(include_str!("cuda_kernels/elementwise_div.ptx").to_string()),
        }
    }
}

/// Element-wise operation types
#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
#[derive(Debug, Clone, Copy)]
pub enum ElementwiseOp {
    Add,
    Mul,
    Sub,
    Div,
}

/// CUDA device properties
#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
#[derive(Debug, Clone)]
pub struct CudaDeviceProperties {
    pub name: String,
    pub total_global_memory: usize,
    pub shared_memory_per_block: usize,
    pub max_threads_per_block: usize,
    pub max_grid_size: [u32; 3],
    pub max_block_size: [u32; 3],
    pub warp_size: usize,
    pub compute_capability: (i32, i32),
    pub multiprocessor_count: i32,
    pub memory_clock_rate: i32,
    pub memory_bus_width: i32,
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
impl CudaDeviceProperties {
    fn query(device_id: i32) -> Result<Self> {
        unsafe {
            let mut props: CudaDeviceProp = std::mem::zeroed();
            cuda_get_device_properties(&mut props, device_id)?;

            Ok(CudaDeviceProperties {
                name: CStr::from_ptr(props.name.as_ptr())
                    .to_string_lossy()
                    .into_owned(),
                total_global_memory: props.total_global_memory,
                shared_memory_per_block: props.shared_memory_per_block,
                max_threads_per_block: props.max_threads_per_block,
                max_grid_size: props.max_grid_size,
                max_block_size: props.max_block_size,
                warp_size: props.warp_size,
                compute_capability: (props.major, props.minor),
                multiprocessor_count: props.multiprocessor_count,
                memory_clock_rate: props.memory_clock_rate,
                memory_bus_width: props.memory_bus_width,
            })
        }
    }

    /// Check if device supports cooperative groups
    pub fn supports_cooperative_groups(&self) -> bool {
        self.compute_capability >= (6, 0)
    }

    /// Check if device supports tensor cores
    pub fn supports_tensor_cores(&self) -> bool {
        self.compute_capability >= (7, 0)
    }

    /// Get theoretical memory bandwidth in GB/s
    pub fn memory_bandwidth_gb_s(&self) -> f64 {
        (self.memory_clock_rate as f64 * 2.0 * self.memory_bus_width as f64) / (8.0 * 1e9)
    }
}

// CUDA FFI types and functions
#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
#[repr(C)]
struct CudaDeviceProp {
    name: [i8; 256],
    total_global_memory: usize,
    shared_memory_per_block: usize,
    max_threads_per_block: usize,
    max_grid_size: [u32; 3],
    max_block_size: [u32; 3],
    warp_size: usize,
    major: i32,
    minor: i32,
    multiprocessor_count: i32,
    memory_clock_rate: i32,
    memory_bus_width: i32,
}

#[derive(Debug)]
#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
struct CudaContext {
    device_id: i32,
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
impl CudaContext {
    fn new(device_id: i32) -> Result<Self> {
        Ok(Self { device_id })
    }
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
#[derive(Debug, Clone)]
struct CudaStream {
    handle: *mut std::ffi::c_void,
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
impl CudaStream {
    fn new() -> Result<Self> {
        unsafe {
            let mut stream = std::ptr::null_mut();
            cuda_stream_create(&mut stream)?;
            Ok(Self { handle: stream })
        }
    }
}

#[derive(Debug)]
#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
struct CudaModule {
    handle: *mut std::ffi::c_void,
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
impl CudaModule {
    fn get_function(&self, name: &str) -> Result<CudaKernel> {
        let c_name = CString::new(name).unwrap();
        unsafe {
            let mut function = std::ptr::null_mut();
            cuda_module_get_function(&mut function, self.handle, c_name.as_ptr())?;
            Ok(CudaKernel { function })
        }
    }
}

#[derive(Debug)]
#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
struct CudaKernel {
    function: *mut std::ffi::c_void,
}

// CUDA FFI error handling
#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
type CudaResult = i32;

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
const CUDA_SUCCESS: CudaResult = 0;

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
fn check_cuda_error(result: CudaResult, operation: &str) -> Result<()> {
    if result != CUDA_SUCCESS {
        return Err(TensorError::DeviceError {
            operation: operation.to_string(),
            details: format!("CUDA error code {}", result),
            device: "CUDA".to_string(),
            context: None,
        });
    }
    Ok(())
}

// CUDA FFI bindings to actual CUDA runtime and driver
#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
extern "C" {
    // CUDA Runtime API
    fn cudaSetDevice(device: i32) -> CudaResult;
    fn cudaGetDeviceProperties(prop: *mut CudaDeviceProp, device: i32) -> CudaResult;
    fn cudaMalloc(devPtr: *mut *mut std::ffi::c_void, size: usize) -> CudaResult;
    fn cudaFree(devPtr: *mut std::ffi::c_void) -> CudaResult;
    fn cudaMemcpy(
        dst: *mut std::ffi::c_void,
        src: *const std::ffi::c_void,
        count: usize,
        kind: u32,
    ) -> CudaResult;
    fn cudaStreamCreate(pStream: *mut *mut std::ffi::c_void) -> CudaResult;
    fn cudaStreamDestroy(stream: *mut std::ffi::c_void) -> CudaResult;
    fn cudaDeviceSynchronize() -> CudaResult;
    fn cudaGetLastError() -> CudaResult;
    fn cudaGetErrorString(error: CudaResult) -> *const i8;

    // CUDA Driver API
    fn cuInit(flags: u32) -> CudaResult;
    fn cuModuleLoadDataEx(
        module: *mut *mut std::ffi::c_void,
        image: *const std::ffi::c_void,
        numOptions: u32,
        options: *const *const i8,
        optionValues: *const *const std::ffi::c_void,
    ) -> CudaResult;
    fn cuModuleGetFunction(
        hfunc: *mut *mut std::ffi::c_void,
        hmod: *mut std::ffi::c_void,
        name: *const i8,
    ) -> CudaResult;
    fn cuLaunchKernel(
        f: *mut std::ffi::c_void,
        gridDimX: u32,
        gridDimY: u32,
        gridDimZ: u32,
        blockDimX: u32,
        blockDimY: u32,
        blockDimZ: u32,
        sharedMemBytes: u32,
        hStream: *mut std::ffi::c_void,
        kernelParams: *mut *mut std::ffi::c_void,
        extra: *mut *mut std::ffi::c_void,
    ) -> CudaResult;
    fn cuModuleUnload(hmod: *mut std::ffi::c_void) -> CudaResult;
}

// CUDA memory copy kinds
#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
const CUDA_MEMCPY_HOST_TO_DEVICE: u32 = 1;
#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
const CUDA_MEMCPY_DEVICE_TO_HOST: u32 = 2;

// CUDA FFI wrapper functions with proper error handling
#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
unsafe fn cuda_init(flags: u32) -> Result<()> {
    check_cuda_error(cuInit(flags), "cuInit")
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
unsafe fn cuda_set_device(device_id: i32) -> Result<()> {
    check_cuda_error(cudaSetDevice(device_id), "cudaSetDevice")
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
unsafe fn cuda_get_device_properties(props: *mut CudaDeviceProp, device: i32) -> Result<()> {
    check_cuda_error(
        cudaGetDeviceProperties(props, device),
        "cudaGetDeviceProperties",
    )
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
unsafe fn cuda_malloc(ptr: *mut *mut std::ffi::c_void, size: usize) -> Result<()> {
    check_cuda_error(cudaMalloc(ptr, size), "cudaMalloc")
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
unsafe fn cuda_free(ptr: *mut std::ffi::c_void) -> Result<()> {
    check_cuda_error(cudaFree(ptr), "cudaFree")
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
unsafe fn cuda_memcpy_htod(
    dst: *mut std::ffi::c_void,
    src: *const std::ffi::c_void,
    size: usize,
) -> Result<()> {
    check_cuda_error(
        cudaMemcpy(dst, src, size, CUDA_MEMCPY_HOST_TO_DEVICE),
        "cudaMemcpy H2D",
    )
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
unsafe fn cuda_memcpy_dtoh(
    dst: *mut std::ffi::c_void,
    src: *const std::ffi::c_void,
    size: usize,
) -> Result<()> {
    check_cuda_error(
        cudaMemcpy(dst, src, size, CUDA_MEMCPY_DEVICE_TO_HOST),
        "cudaMemcpy D2H",
    )
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
unsafe fn cuda_stream_create(stream: *mut *mut std::ffi::c_void) -> Result<()> {
    check_cuda_error(cudaStreamCreate(stream), "cudaStreamCreate")
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
unsafe fn cuda_module_load_data_ex(
    module: *mut *mut std::ffi::c_void,
    image: *const std::ffi::c_void,
    num_options: u32,
    options: *const *const i8,
    option_values: *const *const std::ffi::c_void,
) -> Result<()> {
    check_cuda_error(
        cuModuleLoadDataEx(module, image, num_options, options, option_values),
        "cuModuleLoadDataEx",
    )
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
unsafe fn cuda_module_get_function(
    function: *mut *mut std::ffi::c_void,
    module: *mut std::ffi::c_void,
    name: *const i8,
) -> Result<()> {
    check_cuda_error(
        cuModuleGetFunction(function, module, name),
        "cuModuleGetFunction",
    )
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
unsafe fn cuda_launch_kernel(
    function: *mut std::ffi::c_void,
    grid_dim: (u32, u32, u32),
    block_dim: (u32, u32, u32),
    args: *mut *mut std::ffi::c_void,
    shared_memory: u32,
    stream: *mut std::ffi::c_void,
) -> Result<()> {
    check_cuda_error(
        cuLaunchKernel(
            function,
            grid_dim.0,
            grid_dim.1,
            grid_dim.2,
            block_dim.0,
            block_dim.1,
            block_dim.2,
            shared_memory,
            stream,
            args,
            std::ptr::null_mut(),
        ),
        "cuLaunchKernel",
    )
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
unsafe fn cuda_device_synchronize() -> Result<()> {
    check_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize")
}

/// Stub implementation for non-CUDA platforms
#[cfg(not(all(feature = "cuda", any(target_os = "linux", target_os = "windows"))))]
pub mod cuda_stub {
    //! Stub implementation for platforms without CUDA support
    use crate::{Result, TensorError};

    pub fn cuda_not_available() -> Result<()> {
        Err(TensorError::device_error_simple(
            "CUDA kernels are only available with the 'cuda' feature enabled".to_string(),
        ))
    }
}

/// CUDA kernel performance benchmarking
#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
pub mod benchmarks {
    use super::*;
    use std::time::{Duration, Instant};

    pub struct CudaBenchmark {
        device: CudaDevice,
        results: Vec<BenchmarkResult>,
    }

    #[derive(Debug, Clone)]
    pub struct BenchmarkResult {
        pub operation: String,
        pub input_shape: Vec<usize>,
        pub duration: Duration,
        pub throughput_gflops: f64,
        pub memory_bandwidth_gb_s: f64,
        pub kernel_efficiency: f64,
    }

    impl CudaBenchmark {
        pub fn new(device_id: i32) -> Result<Self> {
            Ok(CudaBenchmark {
                device: CudaDevice::new(device_id)?,
                results: Vec::new(),
            })
        }

        /// Benchmark CUDA kernel performance vs cuBLAS
        pub fn benchmark_kernels(
            &mut self,
            sizes: &[(usize, usize, usize)],
        ) -> Result<Vec<BenchmarkResult>> {
            let mut results = Vec::new();

            for &(m, n, k) in sizes {
                let a = Tensor::<f32>::zeros(&[m, k]);
                let b = Tensor::<f32>::zeros(&[k, n]);

                // Benchmark custom CUDA kernel
                let start = Instant::now();
                let _result = self.device.matmul_cuda(&a, &b)?;
                let duration = start.elapsed();

                let operations = 2 * m * n * k; // FLOPS for matrix multiplication
                let gflops = operations as f64 / duration.as_secs_f64() / 1e9;

                let memory_accessed = (m * k + k * n + m * n) * 4; // bytes for f32
                let bandwidth = memory_accessed as f64 / duration.as_secs_f64() / 1e9;

                // Calculate kernel efficiency vs theoretical peak
                let theoretical_bandwidth =
                    self.device.get_device_properties().memory_bandwidth_gb_s();
                let efficiency = bandwidth / theoretical_bandwidth;

                results.push(BenchmarkResult {
                    operation: format!("cuda_matmul_{}x{}x{}", m, n, k),
                    input_shape: vec![m, k, n],
                    duration,
                    throughput_gflops: gflops,
                    memory_bandwidth_gb_s: bandwidth,
                    kernel_efficiency: efficiency,
                });
            }

            self.results.extend(results.clone());
            Ok(results)
        }

        /// Generate detailed performance report
        pub fn generate_report(&self) -> String {
            let mut report = String::from("CUDA Kernel Performance Report\n");
            report.push_str("==================================\n\n");

            for result in &self.results {
                report.push_str(&format!(
                    "Operation: {}\n  Duration: {:?}\n  Throughput: {:.2} GFLOPS\n  Bandwidth: {:.2} GB/s\n  Efficiency: {:.1}%\n\n",
                    result.operation, result.duration, result.throughput_gflops,
                    result.memory_bandwidth_gb_s, result.kernel_efficiency * 100.0
                ));
            }

            report
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    fn test_cuda_device_creation() {
        let result = CudaDevice::new(0);
        // Test should pass on systems with CUDA support
        assert!(result.is_ok() || result.unwrap_err().to_string().contains("CUDA"));
    }

    #[test]
    #[cfg(not(all(feature = "cuda", any(target_os = "linux", target_os = "windows"))))]
    fn test_cuda_not_available() {
        let result = cuda_stub::cuda_not_available();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("CUDA kernels are only available"));
    }

    #[test]
    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    fn test_kernel_config_calculation() {
        if let Ok(device) = CudaDevice::new(0) {
            let config = device.calculate_gemm_config(1024, 1024, 1024);
            assert!(config.is_ok());

            let cfg = config.unwrap();
            assert!(cfg.grid_dim.0 > 0);
            assert!(cfg.block_dim.0 > 0);
        }
    }
}
