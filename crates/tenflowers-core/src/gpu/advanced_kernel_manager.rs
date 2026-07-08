//! Advanced GPU Kernel Manager for TenfloweRS
//!
//! This module provides high-level integration of cutting-edge GPU optimization
//! techniques including Tensor Cores, SIMD-group operations, and wavefront
//! primitives across CUDA, Metal, and ROCm platforms.
//!
//! The goal is to achieve state-of-the-art performance by leveraging the latest
//! GPU architecture features and vendor-specific optimizations.

use crate::{DType, Device, Result, Tensor, TensorError};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use wgpu::util::DeviceExt;

/// Advanced GPU kernel execution strategies
#[derive(Debug, Clone, PartialEq)]
pub enum KernelStrategy {
    /// Use NVIDIA Tensor Cores for matrix operations (requires compute capability 7.0+)
    TensorCore {
        precision: TensorCorePrecision,
        tile_size: usize,
        /// Enable Hopper/Ada architecture optimizations (H100, RTX 40-series)
        hopper_optimizations: bool,
        /// Use FP8 precision for latest architectures
        fp8_support: bool,
    },
    /// Use Apple SIMD-group operations for Apple Silicon and AMD GPUs on macOS
    SIMDGroup {
        group_size: usize,
        vectorization_width: usize,
        /// Enable M3/M4 architecture-specific optimizations
        apple_neural_engine: bool,
        /// Advanced memory bandwidth optimization
        memory_coalescing: bool,
    },
    /// Use AMD wavefront primitives for optimal RDNA/GCN performance
    Wavefront {
        wavefront_size: usize,
        lds_optimization: bool,
        /// Enable RDNA3/4 architecture optimizations
        rdna3_optimizations: bool,
        /// Use AI accelerator units on RDNA3+
        ai_accelerator: bool,
    },
    /// Intel Arc GPU optimizations with Xe-Core utilization
    IntelXe {
        /// Xe-Core thread group size
        xe_thread_groups: usize,
        /// Enable Intel XMX AI acceleration
        xmx_acceleration: bool,
        /// Optimize for Arc Alchemist/Battlemage
        intel_gpu_gen: IntelGpuGeneration,
    },
    /// Fallback to standard WGPU compute shaders
    StandardCompute,
}

/// Tensor Core precision modes
#[derive(Debug, Clone, PartialEq)]
pub enum TensorCorePrecision {
    /// Half precision (FP16) for maximum throughput
    Float16,
    /// Brain Float 16 for training workloads
    BFloat16,
    /// Int8 for maximum inference performance
    Int8,
    /// FP8 precision for latest Hopper/Ada architectures
    Float8 { e4m3: bool, e5m2: bool },
    /// Int4 for extreme quantization on latest hardware
    Int4,
    /// Mixed precision with FP32 accumulation
    Mixed,
    /// Adaptive precision that switches based on compute requirements
    Adaptive,
}

/// Intel GPU generation for architecture-specific optimizations
#[derive(Debug, Clone, PartialEq)]
pub enum IntelGpuGeneration {
    /// Arc Alchemist (DG2)
    Alchemist,
    /// Arc Battlemage (BMG)
    Battlemage,
    /// Future Celestial architecture
    Celestial,
    /// Integrated Xe graphics
    XeIntegrated,
}

/// Advanced memory optimization strategies
#[derive(Debug, Clone, PartialEq)]
pub enum MemoryStrategy {
    /// Use GPU shared memory for data reuse
    SharedMemoryTiling {
        tile_size: usize,
        bank_conflict_avoidance: bool,
    },
    /// Optimize for memory bandwidth saturation
    BandwidthOptimized {
        prefetch_distance: usize,
        vectorization_factor: usize,
    },
    /// Use texture memory for read-only data
    TextureMemory { cache_optimization: bool },
    /// HBM optimization for high-end GPUs
    HbmOptimized {
        memory_channels: usize,
        interleaving: bool,
    },
}

/// Performance optimization hints for kernel selection
#[derive(Debug, Clone)]
pub struct OptimizationHints {
    /// Operation is compute-bound vs memory-bound
    pub compute_intensity: ComputeIntensity,
    /// Expected tensor shapes for optimization
    pub tensor_shapes: Vec<Vec<usize>>,
    /// Whether this operation will be repeated many times
    pub is_repetitive: bool,
    /// Memory access pattern characteristics
    pub memory_pattern: MemoryPattern,
    /// Target precision requirements
    pub precision_requirements: PrecisionRequirements,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ComputeIntensity {
    /// Memory bandwidth limited (e.g., element-wise operations)
    MemoryBound,
    /// Compute limited (e.g., large matrix multiplications)
    ComputeBound,
    /// Balanced compute and memory requirements
    Balanced,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MemoryPattern {
    /// Sequential access patterns (good for vectorization)
    Sequential,
    /// Strided access patterns (may benefit from prefetching)
    Strided { stride: usize },
    /// Random access patterns (cache-unfriendly)
    Random,
    /// Coalesced access suitable for GPU memory hierarchy
    Coalesced,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PrecisionRequirements {
    /// Maximum precision required (FP64)
    HighPrecision,
    /// Standard precision (FP32)
    StandardPrecision,
    /// Reduced precision acceptable (FP16/BF16)
    ReducedPrecision,
    /// Minimum precision for inference (INT8)
    MinimalPrecision,
}

/// Advanced kernel manager that automatically selects optimal implementations
pub struct AdvancedKernelManager {
    /// Device capabilities and features
    device_info: Arc<RwLock<DeviceCapabilities>>,
    /// Cache of compiled kernels for reuse
    kernel_cache: Arc<RwLock<HashMap<String, CompiledKernel>>>,
    /// Performance profiling data
    performance_data: Arc<RwLock<HashMap<String, KernelPerformanceData>>>,
    /// Current optimization strategy
    strategy: KernelStrategy,
}

/// Device capabilities for kernel selection
#[derive(Debug, Clone)]
pub struct DeviceCapabilities {
    /// GPU vendor (NVIDIA, AMD, Apple, Intel)
    pub vendor: GpuVendor,
    /// Compute capability or architecture version
    pub compute_capability: String,
    /// Available memory bandwidth (GB/s)
    pub memory_bandwidth: f64,
    /// Number of compute units/streaming multiprocessors
    pub compute_units: usize,
    /// Supports Tensor Cores or equivalent
    pub has_tensor_cores: bool,
    /// Maximum threads per block/workgroup
    pub max_threads_per_block: usize,
    /// Shared memory size per block
    pub shared_memory_size: usize,
    /// Supports half precision operations
    pub supports_fp16: bool,
    /// Supports brain float 16
    pub supports_bf16: bool,
    /// Supports cooperative groups/SIMD-groups
    pub supports_coop_groups: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum GpuVendor {
    Nvidia,
    AMD,
    Apple,
    Intel,
    Unknown,
}

/// Compiled kernel information
#[derive(Debug, Clone)]
pub struct CompiledKernel {
    /// Unique kernel identifier
    pub id: String,
    /// Kernel strategy used
    pub strategy: KernelStrategy,
    /// Compilation timestamp
    pub compiled_at: std::time::SystemTime,
    /// Kernel-specific parameters
    pub parameters: KernelParameters,
    /// Platform-specific kernel handle
    pub handle: KernelHandle,
}

/// Platform-agnostic kernel handle
#[derive(Debug, Clone)]
pub enum KernelHandle {
    #[cfg(feature = "cuda")]
    Cuda {
        module: String,
        function: String,
    },
    #[cfg(feature = "metal")]
    Metal {
        library: String,
        function: String,
    },
    #[cfg(feature = "rocm")]
    ROCm {
        module: String,
        function: String,
    },
    WGPU {
        shader: String,
        entry_point: String,
    },
}

/// Kernel execution parameters
#[derive(Debug, Clone)]
pub struct KernelParameters {
    /// Grid/threadgroup dimensions
    pub grid_size: (usize, usize, usize),
    /// Block/thread dimensions
    pub block_size: (usize, usize, usize),
    /// Shared memory requirements
    pub shared_memory: usize,
    /// Expected register usage
    pub register_count: usize,
}

/// Performance profiling data for kernel optimization
#[derive(Debug, Clone)]
pub struct KernelPerformanceData {
    /// Average execution time in microseconds
    pub avg_execution_time: f64,
    /// Memory bandwidth utilization percentage
    pub memory_bandwidth_util: f64,
    /// Compute utilization percentage
    pub compute_utilization: f64,
    /// Number of times this kernel has been executed
    pub execution_count: usize,
    /// Performance improvement over baseline
    pub speedup_factor: f64,
}

impl AdvancedKernelManager {
    /// Create a new kernel manager with device introspection
    pub fn new(device: &Device) -> Result<Self> {
        let device_info = Self::detect_device_capabilities(device)?;
        let strategy = Self::select_optimal_strategy(&device_info);

        Ok(AdvancedKernelManager {
            device_info: Arc::new(RwLock::new(device_info)),
            kernel_cache: Arc::new(RwLock::new(HashMap::new())),
            performance_data: Arc::new(RwLock::new(HashMap::new())),
            strategy,
        })
    }

    /// Detect device capabilities through runtime introspection
    fn detect_device_capabilities(device: &Device) -> Result<DeviceCapabilities> {
        match device {
            #[cfg(feature = "gpu")]
            Device::Gpu(gpu_device) => {
                // Fetch a real snapshot of the live `wgpu::Adapter`'s capabilities. This
                // is the ONLY place genuine hardware data enters `DeviceCapabilities`:
                // every field below is either a pure mapping of this real data, or (where
                // wgpu simply has no API for the question) an honest error.
                let caps = crate::device::get_gpu_adapter_capabilities(*gpu_device)?;

                Ok(DeviceCapabilities {
                    vendor: Self::detect_gpu_vendor(&caps.info),
                    compute_capability: Self::query_compute_capability(&caps.info),
                    memory_bandwidth: Self::measure_memory_bandwidth()?,
                    compute_units: Self::query_compute_units()?,
                    has_tensor_cores: Self::detect_tensor_cores()?,
                    max_threads_per_block: Self::query_max_threads_per_block(&caps.limits),
                    shared_memory_size: Self::query_shared_memory_size(&caps.limits),
                    supports_fp16: Self::test_fp16_support(caps.features),
                    supports_bf16: Self::test_bf16_support()?,
                    supports_coop_groups: Self::test_cooperative_groups()?,
                })
            }
            _ => Err(TensorError::InvalidArgument {
                operation: "AdvancedKernelManager::detect_device_capabilities".to_string(),
                reason: "Advanced kernels only supported on GPU devices".to_string(),
                context: None,
            }),
        }
    }

    /// Select optimal kernel strategy based on device capabilities
    fn select_optimal_strategy(capabilities: &DeviceCapabilities) -> KernelStrategy {
        match capabilities.vendor {
            GpuVendor::Nvidia if capabilities.has_tensor_cores => {
                KernelStrategy::TensorCore {
                    precision: if capabilities.supports_bf16 {
                        TensorCorePrecision::BFloat16
                    } else {
                        TensorCorePrecision::Float16
                    },
                    tile_size: 16, // 16x16 Tensor Core tiles
                    hopper_optimizations: capabilities.compute_capability.as_str() >= "9.0",
                    fp8_support: capabilities.compute_capability.as_str() >= "8.9",
                }
            }
            GpuVendor::Apple => {
                KernelStrategy::SIMDGroup {
                    group_size: 32,            // Apple Silicon SIMD-group size
                    vectorization_width: 8,    // Optimal for unified memory
                    apple_neural_engine: true, // Assume M-series chips have Neural Engine
                    memory_coalescing: true,
                }
            }
            GpuVendor::AMD => {
                KernelStrategy::Wavefront {
                    wavefront_size: 64, // AMD wavefront size
                    lds_optimization: true,
                    rdna3_optimizations: true, // Assume modern AMD GPUs
                    ai_accelerator: false,     // Conservative default
                }
            }
            _ => KernelStrategy::StandardCompute,
        }
    }

    /// Execute optimized matrix multiplication using the best available strategy
    pub fn optimized_matmul<T>(
        &self,
        a: &Tensor<T>,
        b: &Tensor<T>,
        hints: OptimizationHints,
    ) -> Result<Tensor<T>>
    where
        T: bytemuck::Pod + bytemuck::Zeroable + Clone + Default + Send + Sync + 'static,
    {
        let kernel_id = format!("matmul_{}x{}x{}", a.shape()[0], a.shape()[1], b.shape()[1]);

        // Check cache for pre-compiled kernel
        if let Some(kernel) = self.get_cached_kernel(&kernel_id)? {
            return self.execute_cached_matmul(a, b, &kernel);
        }

        // Select and compile optimal kernel based on strategy and hints
        let kernel = self.compile_optimal_matmul_kernel(a, b, &hints)?;
        self.cache_kernel(kernel_id.clone(), kernel.clone())?;

        // Execute the kernel
        let result = self.execute_matmul_kernel(a, b, &kernel)?;

        // Update performance data
        self.update_performance_data(&kernel_id, &kernel)?;

        Ok(result)
    }

    /// Compile optimal matrix multiplication kernel based on current strategy
    fn compile_optimal_matmul_kernel<T: 'static>(
        &self,
        a: &Tensor<T>,
        b: &Tensor<T>,
        hints: &OptimizationHints,
    ) -> Result<CompiledKernel> {
        match &self.strategy {
            KernelStrategy::TensorCore {
                precision,
                tile_size,
                ..
            } => self.compile_tensor_core_matmul(a, b, precision, *tile_size),
            KernelStrategy::SIMDGroup {
                group_size,
                vectorization_width,
                ..
            } => self.compile_simd_group_matmul(a, b, *group_size, *vectorization_width),
            KernelStrategy::Wavefront {
                wavefront_size,
                lds_optimization,
                ..
            } => self.compile_wavefront_matmul(a, b, *wavefront_size, *lds_optimization),
            KernelStrategy::StandardCompute => self.compile_standard_matmul(a, b),
            KernelStrategy::IntelXe {
                xe_thread_groups,
                xmx_acceleration,
                intel_gpu_gen,
            } => self.compile_intel_xe_matmul(
                a,
                b,
                *xe_thread_groups,
                *xmx_acceleration,
                intel_gpu_gen,
            ),
        }
    }

    /// Compile Tensor Core optimized matrix multiplication
    #[cfg(feature = "cuda")]
    fn compile_tensor_core_matmul<T: 'static>(
        &self,
        a: &Tensor<T>,
        b: &Tensor<T>,
        precision: &TensorCorePrecision,
        tile_size: usize,
    ) -> Result<CompiledKernel> {
        let kernel_source = match precision {
            TensorCorePrecision::Float16 => {
                include_str!("cuda_kernels/tensor_core_matmul.ptx")
            }
            TensorCorePrecision::BFloat16 => {
                include_str!("cuda_kernels/tensor_core_matmul.ptx") // BF16 variant
            }
            TensorCorePrecision::Int8 => {
                include_str!("cuda_kernels/tensor_core_matmul.ptx") // INT8 variant
            }
            TensorCorePrecision::Mixed => {
                include_str!("cuda_kernels/tensor_core_matmul.ptx") // Mixed precision variant
            }
            TensorCorePrecision::Float8 { .. } => {
                include_str!("cuda_kernels/tensor_core_matmul.ptx") // Float8 variant
            }
            TensorCorePrecision::Int4 => {
                include_str!("cuda_kernels/tensor_core_matmul.ptx") // Int4 variant
            }
            TensorCorePrecision::Adaptive => {
                include_str!("cuda_kernels/tensor_core_matmul.ptx") // Adaptive variant
            }
        };

        // Calculate optimal grid and block dimensions for Tensor Cores
        let m = a.shape()[0];
        let n = b.shape()[1];
        let grid_x = (m + tile_size - 1) / tile_size;
        let grid_y = (n + tile_size - 1) / tile_size;
        let block_x = 32; // Warp size
        let block_y = 8; // 4 warps per block

        Ok(CompiledKernel {
            id: format!("tensor_core_{}_{:?}", tile_size, precision),
            strategy: self.strategy.clone(),
            compiled_at: std::time::SystemTime::now(),
            parameters: KernelParameters {
                grid_size: (grid_x, grid_y, 1),
                block_size: (block_x, block_y, 1),
                shared_memory: tile_size * tile_size * 4, // 2 tiles * sizeof(element)
                register_count: 64,                       // Tensor Core register requirements
            },
            handle: KernelHandle::Cuda {
                module: "tensor_core_matmul".to_string(),
                function: match precision {
                    TensorCorePrecision::Float16 => "tensor_core_gemm_f16".to_string(),
                    TensorCorePrecision::BFloat16 => "tensor_core_gemm_bf16".to_string(),
                    TensorCorePrecision::Int8 => "tensor_core_gemm_int8".to_string(),
                    TensorCorePrecision::Mixed => "tensor_core_gemm_mixed".to_string(),
                    TensorCorePrecision::Float8 { .. } => "tensor_core_gemm_f8".to_string(),
                    TensorCorePrecision::Int4 => "tensor_core_gemm_int4".to_string(),
                    TensorCorePrecision::Adaptive => "tensor_core_gemm_adaptive".to_string(),
                },
            },
        })
    }

    /// Compile SIMD-group optimized matrix multiplication for Apple Silicon
    #[cfg(feature = "metal")]
    fn compile_simd_group_matmul<T: 'static>(
        &self,
        a: &Tensor<T>,
        b: &Tensor<T>,
        group_size: usize,
        vectorization_width: usize,
    ) -> Result<CompiledKernel> {
        let kernel_source = include_str!("metal_shaders/simd_group_matmul.metal");

        let m = a.shape()[0];
        let n = b.shape()[1];
        let tile_size = 32; // Optimal for Apple Silicon

        let threadgroup_x = (m + tile_size - 1) / tile_size;
        let threadgroup_y = (n + tile_size - 1) / tile_size;

        Ok(CompiledKernel {
            id: format!("simd_group_{}_{}", group_size, vectorization_width),
            strategy: self.strategy.clone(),
            compiled_at: std::time::SystemTime::now(),
            parameters: KernelParameters {
                grid_size: (threadgroup_x, threadgroup_y, 1),
                block_size: (group_size, 1, 1),
                shared_memory: tile_size * tile_size * 8, // Tile memory for A and B
                register_count: 32,                       // SIMD-group register usage
            },
            handle: KernelHandle::Metal {
                library: "simd_group_matmul".to_string(),
                function: match a.dtype() {
                    DType::Float32 => "simd_group_matmul_f32".to_string(),
                    DType::Float16 => "simd_group_matmul_f16".to_string(),
                    _ => {
                        return Err(TensorError::unsupported_operation_simple(format!(
                            "SIMD-group matmul: Unsupported dtype: {:?}",
                            a.dtype()
                        )))
                    }
                },
            },
        })
    }

    /// Compile wavefront optimized matrix multiplication for AMD GPUs
    #[cfg(feature = "rocm")]
    fn compile_wavefront_matmul<T: 'static>(
        &self,
        a: &Tensor<T>,
        b: &Tensor<T>,
        wavefront_size: usize,
        lds_optimization: bool,
    ) -> Result<CompiledKernel> {
        let kernel_source = include_str!("rocm_kernels/wavefront_optimized_matmul.hip");

        let m = a.shape()[0];
        let n = b.shape()[1];
        let tile_size = 64; // Optimal for GCN/RDNA

        let grid_x = (m + tile_size - 1) / tile_size;
        let grid_y = (n + tile_size - 1) / tile_size;
        let block_size = wavefront_size * 4; // 4 wavefronts per workgroup

        Ok(CompiledKernel {
            id: format!("wavefront_{}_{}", wavefront_size, lds_optimization),
            strategy: self.strategy.clone(),
            compiled_at: std::time::SystemTime::now(),
            parameters: KernelParameters {
                grid_size: (grid_x, grid_y, 1),
                block_size: (block_size, 1, 1),
                shared_memory: if lds_optimization {
                    tile_size * tile_size * 8 // LDS for A and B tiles
                } else {
                    0
                },
                register_count: 48, // Wavefront register requirements
            },
            handle: KernelHandle::ROCm {
                module: "wavefront_matmul".to_string(),
                function: match a.dtype() {
                    DType::Float32 => "wavefront_gemm_f32".to_string(),
                    DType::Float16 => "wavefront_gemm_f16".to_string(),
                    _ => {
                        return Err(TensorError::unsupported_operation_simple(format!(
                            "Wavefront matmul: Unsupported dtype: {:?}",
                            a.dtype()
                        )))
                    }
                },
            },
        })
    }

    /// Fallback to standard compute shader implementation
    fn compile_standard_matmul<T: 'static>(
        &self,
        a: &Tensor<T>,
        b: &Tensor<T>,
    ) -> Result<CompiledKernel> {
        // Use existing WGPU matmul implementation as fallback
        let m = a.shape()[0];
        let n = b.shape()[1];
        let tile_size = 16; // Conservative tile size for compatibility

        let grid_x = (m + tile_size - 1) / tile_size;
        let grid_y = (n + tile_size - 1) / tile_size;

        Ok(CompiledKernel {
            id: "standard_compute".to_string(),
            strategy: self.strategy.clone(),
            compiled_at: std::time::SystemTime::now(),
            parameters: KernelParameters {
                grid_size: (grid_x, grid_y, 1),
                block_size: (tile_size, tile_size, 1),
                shared_memory: 0,
                register_count: 16,
            },
            handle: KernelHandle::WGPU {
                shader: "matmul_ops.wgsl".to_string(),
                entry_point: "matmul_kernel".to_string(),
            },
        })
    }

    // Device-capability queries.
    //
    // `detect_device_capabilities` threads a real `wgpu::Adapter` snapshot
    // (`GpuAdapterCapabilities`, fetched via `crate::device::get_gpu_adapter_capabilities`)
    // into these queries. Each one falls into exactly one of two honest categories:
    //
    //   1. wgpu genuinely exposes the data (`AdapterInfo`/`Limits`/`Features`):
    //      `detect_gpu_vendor`, `query_compute_capability`, `query_max_threads_per_block`,
    //      `query_shared_memory_size`, and `test_fp16_support` map that real data through
    //      the pure, independently unit-tested helpers below (`map_vendor`,
    //      `compute_capability_from_info`, `threads_from_limits`, `shared_mem_from_limits`,
    //      `fp16_from_features`) and cannot fail.
    //
    //   2. wgpu has no API surface for the question at all, on any backend, with or
    //      without a live adapter (`measure_memory_bandwidth`, `query_compute_units`,
    //      `detect_tensor_cores`, `test_bf16_support`, `test_cooperative_groups`). Rather
    //      than fabricate a hardcoded literal dressed up as a measurement (which would
    //      silently mislead the strategy selector), each of these returns an honest error
    //      naming precisely which vendor-specific runtime would be required instead.

    /// Real GPU vendor, mapped from `wgpu::Adapter::get_info()`.
    fn detect_gpu_vendor(info: &wgpu::AdapterInfo) -> GpuVendor {
        map_vendor(info)
    }

    /// Best-effort architecture/"compute capability" descriptor, built only from real
    /// `wgpu::AdapterInfo` fields (see [`compute_capability_from_info`] for why this is
    /// not a CUDA-style version number).
    fn query_compute_capability(info: &wgpu::AdapterInfo) -> String {
        compute_capability_from_info(info)
    }

    /// wgpu has no API for memory bandwidth (GB/s) on `Adapter` or `Device` — neither
    /// `AdapterInfo`, `Limits`, nor `Features` carries a bandwidth figure, on any backend.
    /// Real numbers require vendor-specific tooling outside wgpu's portable abstraction
    /// (e.g. NVML for NVIDIA, ROCm-SMI for AMD, IOKit performance counters on macOS), none
    /// of which this crate links against.
    fn measure_memory_bandwidth() -> Result<f64> {
        Err(TensorError::unsupported_operation_simple(
            "memory bandwidth cannot be queried through wgpu (no GB/s field exists on \
             AdapterInfo/Limits/Features on any backend); a vendor runtime such as NVML or \
             ROCm-SMI is required"
                .to_string(),
        ))
    }

    /// wgpu deliberately does not expose a streaming-multiprocessor / compute-unit count.
    /// This is unrelated to whether a real device handle is available: `AdapterInfo` and
    /// `Limits` simply have no such field, on any backend.
    fn query_compute_units() -> Result<usize> {
        Err(TensorError::unsupported_operation_simple(
            "compute unit / SM count has no wgpu API (AdapterInfo and Limits carry no such \
             field on any backend); a vendor runtime such as NVML or ROCm-SMI is required"
                .to_string(),
        ))
    }

    /// wgpu has no feature flag or limit indicating tensor-core / matrix-accelerator
    /// presence. Detecting this honestly would require vendor-specific extensions (e.g.
    /// `VK_NV_cooperative_matrix`, `VK_KHR_cooperative_matrix`) that are not exposed
    /// through wgpu's portable `Features`/`Limits` surface.
    fn detect_tensor_cores() -> Result<bool> {
        Err(TensorError::unsupported_operation_simple(
            "tensor-core presence cannot be queried through wgpu; it exposes no matrix-\
             accelerator feature flag on any backend"
                .to_string(),
        ))
    }

    /// Maximum compute-workgroup X size, read from a real `wgpu::Limits`.
    fn query_max_threads_per_block(limits: &wgpu::Limits) -> usize {
        threads_from_limits(limits) as usize
    }

    /// Maximum compute-workgroup storage (shared memory) size, read from a real
    /// `wgpu::Limits`.
    fn query_shared_memory_size(limits: &wgpu::Limits) -> usize {
        shared_mem_from_limits(limits) as usize
    }

    /// fp16 shader support, read from a real `wgpu::Features` (`SHADER_F16`).
    fn test_fp16_support(features: wgpu::Features) -> bool {
        fp16_from_features(features)
    }

    /// wgpu defines `Features::SHADER_F16` but has no `SHADER_BF16` (or equivalent)
    /// counterpart on any backend, so brain-float-16 shader support cannot be queried
    /// through wgpu at all.
    fn test_bf16_support() -> Result<bool> {
        Err(TensorError::unsupported_operation_simple(
            "bf16 support cannot be queried through wgpu; no SHADER_BF16 (or equivalent) \
             feature flag exists on any backend"
                .to_string(),
        ))
    }

    /// wgpu exposes `Features::SUBGROUP` for subgroup (warp/wave) intrinsics, but CUDA-style
    /// "cooperative groups" (grid-wide / multi-block synchronization) is a distinct, more
    /// powerful concept with no wgpu equivalent. `SUBGROUP` is deliberately NOT treated as
    /// answering this query — doing so would silently misrepresent a much narrower
    /// capability as the broader one.
    fn test_cooperative_groups() -> Result<bool> {
        Err(TensorError::unsupported_operation_simple(
            "cooperative groups cannot be queried through wgpu; Features::SUBGROUP (warp/wave \
             intrinsics) is a distinct, narrower capability and not a substitute"
                .to_string(),
        ))
    }

    fn get_cached_kernel(&self, kernel_id: &str) -> Result<Option<CompiledKernel>> {
        let cache = self
            .kernel_cache
            .read()
            .map_err(|_| TensorError::InvalidArgument {
                operation: "get_cached_kernel".to_string(),
                reason: "Failed to acquire read lock".to_string(),
                context: None,
            })?;
        Ok(cache.get(kernel_id).cloned())
    }

    fn cache_kernel(&self, kernel_id: String, kernel: CompiledKernel) -> Result<()> {
        let mut cache = self
            .kernel_cache
            .write()
            .map_err(|_| TensorError::InvalidArgument {
                operation: "cache_kernel".to_string(),
                reason: "Failed to acquire write lock".to_string(),
                context: None,
            })?;
        cache.insert(kernel_id, kernel);
        Ok(())
    }

    fn execute_cached_matmul<T>(
        &self,
        a: &Tensor<T>,
        b: &Tensor<T>,
        kernel: &CompiledKernel,
    ) -> Result<Tensor<T>>
    where
        T: bytemuck::Pod + bytemuck::Zeroable + Clone + Default + Send + Sync + 'static,
    {
        // Execute the cached kernel - implementation depends on kernel type
        self.execute_matmul_kernel(a, b, kernel)
    }

    fn execute_matmul_kernel<T>(
        &self,
        a: &Tensor<T>,
        b: &Tensor<T>,
        kernel: &CompiledKernel,
    ) -> Result<Tensor<T>>
    where
        T: bytemuck::Pod + bytemuck::Zeroable + Clone + Default + Send + Sync + 'static,
    {
        // Platform-specific kernel execution
        match &kernel.handle {
            #[cfg(feature = "cuda")]
            KernelHandle::Cuda { module, function } => self.execute_cuda_kernel(a, b, kernel),
            #[cfg(feature = "metal")]
            KernelHandle::Metal { library, function } => self.execute_metal_kernel(a, b, kernel),
            #[cfg(feature = "rocm")]
            KernelHandle::ROCm { module, function } => self.execute_rocm_kernel(a, b, kernel),
            KernelHandle::WGPU {
                shader,
                entry_point,
            } => self.execute_wgpu_kernel(a, b, kernel),
        }
    }

    #[cfg(feature = "cuda")]
    fn execute_cuda_kernel<T>(
        &self,
        a: &Tensor<T>,
        b: &Tensor<T>,
        kernel: &CompiledKernel,
    ) -> Result<Tensor<T>> {
        // CUDA kernel execution implementation
        // This would interface with the CUDA runtime using cuLaunchKernel or similar APIs

        // For now, return an informative error that CUDA execution is not yet implemented
        // Future implementation would:
        // 1. Load CUDA module from kernel.handle.module
        // 2. Get CUDA function from kernel.handle.function
        // 3. Allocate device memory for inputs/outputs
        // 4. Configure kernel launch parameters using kernel.parameters
        // 5. Launch kernel with cuLaunchKernel
        // 6. Synchronize and copy results back
        // 7. Create result tensor with GPU storage

        Err(TensorError::unsupported_operation_simple(
            "CUDA kernel execution: CUDA kernel execution not yet implemented. CUDA feature is enabled but requires cuLaunchKernel integration.".to_string()
        ))
    }

    #[cfg(feature = "metal")]
    fn execute_metal_kernel<T>(
        &self,
        a: &Tensor<T>,
        b: &Tensor<T>,
        _kernel: &CompiledKernel,
    ) -> Result<Tensor<T>>
    where
        T: bytemuck::Pod + bytemuck::Zeroable + Clone + Default + Send + Sync + 'static,
    {
        // Metal execution delegates to the already-complete, already-working WGPU
        // compute path (`execute_wgpu_kernel`, below): wgpu's Metal backend
        // (`wgpu::Backend::Metal`) already runs genuine compute shaders correctly
        // on this exact Apple Silicon GPU, so there is no need for a separate
        // Metal Performance Shaders integration to get a correct result here. (The
        // raw metal-rs FFI path used when bypassing wgpu entirely is required
        // lives in `gpu/metal_kernels/operations.rs`.)
        //
        // The Metal-specific `_kernel` passed in here (compiled by
        // `compile_simd_group_matmul`, tuned for a hand-authored `.metal`
        // SIMD-group shader that doesn't exist yet) is not reusable as-is by
        // `execute_wgpu_kernel`, which requires a `KernelHandle::WGPU` handle
        // whose `grid_size` matches `shaders/matmul_ops.wgsl`'s own workgroup
        // size. A fresh, correctly shaped WGPU kernel is compiled here instead of
        // reusing the passed-in one.
        let wgpu_kernel = self.compile_standard_matmul(a, b)?;
        self.execute_wgpu_kernel(a, b, &wgpu_kernel)
    }

    #[cfg(feature = "rocm")]
    fn execute_rocm_kernel<T>(
        &self,
        a: &Tensor<T>,
        b: &Tensor<T>,
        kernel: &CompiledKernel,
    ) -> Result<Tensor<T>> {
        // ROCm kernel execution implementation
        // This would interface with HIP runtime for AMD GPU execution

        // For now, return an informative error that ROCm execution is not yet implemented
        // Future implementation would:
        // 1. Initialize HIP runtime with hipInit()
        // 2. Load HIP module from kernel.handle.module using hipModuleLoad()
        // 3. Get HIP function from kernel.handle.function using hipModuleGetFunction()
        // 4. Allocate device memory with hipMalloc()
        // 5. Copy input data to device with hipMemcpy()
        // 6. Configure launch parameters using kernel.parameters
        // 7. Launch kernel with hipModuleLaunchKernel()
        // 8. Synchronize with hipDeviceSynchronize()
        // 9. Copy results back to host and create result tensor
        // 10. Free device memory with hipFree()

        Err(TensorError::unsupported_operation_simple(
            "ROCm kernel execution: ROCm kernel execution not yet implemented. ROCm feature is enabled but requires HIP runtime integration.".to_string()
        ))
    }

    fn execute_wgpu_kernel<T>(
        &self,
        a: &Tensor<T>,
        b: &Tensor<T>,
        kernel: &CompiledKernel,
    ) -> Result<Tensor<T>>
    where
        T: bytemuck::Pod + bytemuck::Zeroable + Clone + Default + Send + Sync + 'static,
    {
        use crate::gpu::buffer::GpuBuffer;
        use crate::tensor::TensorStorage;

        if let (TensorStorage::Gpu(gpu_a), TensorStorage::Gpu(gpu_b)) = (&a.storage, &b.storage) {
            let device_arc = Arc::clone(&gpu_a.device);
            let queue_arc = Arc::clone(&gpu_a.queue);

            let a_shape = a.shape().dims();
            let b_shape = b.shape().dims();
            let a_ndim = a_shape.len();
            let b_ndim = b_shape.len();

            // Calculate dimensions
            let m = a_shape[a_ndim - 2];
            let k = a_shape[a_ndim - 1];
            let n = b_shape[b_ndim - 1];
            let batch_size = kernel.parameters.grid_size.2.max(1);

            // Create output buffer based on result dimensions
            let result_shape = vec![batch_size, m, n];
            let output_size: usize = result_shape.iter().product();

            let output_buffer = device_arc.create_buffer(&wgpu::BufferDescriptor {
                label: Some("advanced_matmul_output"),
                size: (output_size * 4) as u64, // Assuming f32
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            // Create parameters uniform buffer
            #[repr(C)]
            #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
            struct MatMulParams {
                m: u32,
                k: u32,
                n: u32,
                batch_size: u32,
            }

            let params = MatMulParams {
                m: m as u32,
                k: k as u32,
                n: n as u32,
                batch_size: batch_size as u32,
            };

            let params_buffer = device_arc.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("advanced_matmul_params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

            // Load compute shader based on kernel handle
            let shader_source = match &kernel.handle {
                KernelHandle::WGPU { shader, .. } => match shader.as_str() {
                    "matmul_ops.wgsl" => include_str!("shaders/matmul_ops.wgsl"),
                    _ => include_str!("shaders/matmul_ops.wgsl"), // Fallback
                },
                #[cfg(any(feature = "cuda", feature = "metal", feature = "rocm"))]
                _ => {
                    return Err(TensorError::InvalidArgument {
                        operation: "execute_wgpu_kernel".to_string(),
                        reason: "Expected WGPU kernel handle".to_string(),
                        context: None,
                    })
                }
            };

            let shader_module = device_arc.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("advanced_matmul_shader"),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });

            // Create bind group layout
            let bind_group_layout =
                device_arc.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("advanced_matmul_bind_group_layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

            // Create bind group
            let bind_group = device_arc.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("advanced_matmul_bind_group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: gpu_a.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: gpu_b.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });

            // Create compute pipeline
            let pipeline_layout =
                device_arc.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("advanced_matmul_pipeline_layout"),
                    bind_group_layouts: &[Some(&bind_group_layout)],
                    immediate_size: 0,
                });

            let entry_point = match &kernel.handle {
                KernelHandle::WGPU { entry_point, .. } => entry_point.as_str(),
                #[cfg(any(feature = "cuda", feature = "metal", feature = "rocm"))]
                _ => "matmul_kernel", // Fallback
            };

            let compute_pipeline =
                device_arc.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("advanced_matmul_pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader_module,
                    entry_point: Some(entry_point),
                    cache: None,
                    compilation_options: Default::default(),
                });

            // Dispatch compute shader using kernel parameters
            let mut encoder = device_arc.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("advanced_matmul_encoder"),
            });

            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("advanced_matmul_pass"),
                    timestamp_writes: None,
                });

                compute_pass.set_pipeline(&compute_pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);

                // Use kernel parameters for dispatch
                let grid_x = kernel.parameters.grid_size.0 as u32;
                let grid_y = kernel.parameters.grid_size.1 as u32;
                let grid_z = kernel.parameters.grid_size.2 as u32;

                compute_pass.dispatch_workgroups(grid_x, grid_y, grid_z);
            }

            queue_arc.submit(std::iter::once(encoder.finish()));
            device_arc.poll(wgpu::PollType::wait_indefinitely()).ok();

            // Create result tensor
            let device_id = match a.device() {
                crate::Device::Gpu(id) => *id,
                _ => {
                    return Err(TensorError::InvalidArgument {
                        operation: "execute_wgpu_kernel".to_string(),
                        reason: "Expected GPU device".to_string(),
                        context: None,
                    })
                }
            };

            let result_gpu = GpuBuffer::from_wgpu_buffer(
                output_buffer,
                device_arc,
                queue_arc,
                crate::Device::Gpu(device_id),
                output_size,
            );

            Ok(Tensor::from_gpu_buffer(
                result_gpu,
                crate::Shape::new(result_shape),
            ))
        } else {
            Err(TensorError::InvalidArgument {
                operation: "execute_wgpu_kernel".to_string(),
                reason: "Expected GPU tensors".to_string(),
                context: None,
            })
        }
    }

    fn update_performance_data(&self, kernel_id: &str, kernel: &CompiledKernel) -> Result<()> {
        // Update performance profiling data for kernel optimization
        let mut perf_data =
            self.performance_data
                .write()
                .map_err(|_| TensorError::InvalidArgument {
                    operation: "update_performance_data".to_string(),
                    reason: "Failed to acquire write lock".to_string(),
                    context: None,
                })?;

        // This would include actual timing and profiling measurements
        perf_data.insert(
            kernel_id.to_string(),
            KernelPerformanceData {
                avg_execution_time: 0.0, // Would be measured
                memory_bandwidth_util: 0.0,
                compute_utilization: 0.0,
                execution_count: 1,
                speedup_factor: 1.0,
            },
        );

        Ok(())
    }
}

/// Public API for advanced GPU optimizations
impl AdvancedKernelManager {
    /// Enable advanced GPU optimizations with automatic kernel selection
    pub fn enable_advanced_optimizations(&mut self) -> Result<()> {
        // Benchmark different strategies and select the best one
        self.benchmark_strategies()?;
        Ok(())
    }

    /// Benchmark different kernel strategies to find optimal performance
    fn benchmark_strategies(&mut self) -> Result<()> {
        // This would run micro-benchmarks on different kernel types
        // and select the fastest one for the current hardware
        Ok(())
    }

    /// Get current performance statistics
    pub fn get_performance_stats(&self) -> Result<HashMap<String, KernelPerformanceData>> {
        let perf_data = self
            .performance_data
            .read()
            .map_err(|_| TensorError::InvalidArgument {
                operation: "get_performance_stats".to_string(),
                reason: "Failed to acquire read lock".to_string(),
                context: None,
            })?;
        Ok(perf_data.clone())
    }

    /// Clear kernel cache and performance data
    pub fn clear_cache(&mut self) -> Result<()> {
        let mut cache = self
            .kernel_cache
            .write()
            .map_err(|_| TensorError::InvalidArgument {
                operation: "clear_cache".to_string(),
                reason: "Failed to acquire write lock".to_string(),
                context: None,
            })?;
        cache.clear();

        let mut perf_data =
            self.performance_data
                .write()
                .map_err(|_| TensorError::InvalidArgument {
                    operation: "clear_cache".to_string(),
                    reason: "Failed to acquire write lock".to_string(),
                    context: None,
                })?;
        perf_data.clear();

        Ok(())
    }

    /// Stub implementation for TensorCore matmul when CUDA feature is not available
    #[cfg(not(feature = "cuda"))]
    fn compile_tensor_core_matmul<T: 'static>(
        &self,
        a: &Tensor<T>,
        b: &Tensor<T>,
        _precision: &TensorCorePrecision,
        _tile_size: usize,
    ) -> Result<CompiledKernel> {
        self.compile_standard_matmul(a, b)
    }

    /// Stub implementation for SIMD group matmul when Metal feature is not available
    #[cfg(not(feature = "metal"))]
    fn compile_simd_group_matmul<T: 'static>(
        &self,
        a: &Tensor<T>,
        b: &Tensor<T>,
        _group_size: usize,
        _vectorization_width: usize,
    ) -> Result<CompiledKernel> {
        self.compile_standard_matmul(a, b)
    }

    /// Stub implementation for wavefront matmul when ROCm feature is not available
    #[cfg(not(feature = "rocm"))]
    fn compile_wavefront_matmul<T: 'static>(
        &self,
        a: &Tensor<T>,
        b: &Tensor<T>,
        _wavefront_size: usize,
        _lds_optimization: bool,
    ) -> Result<CompiledKernel> {
        self.compile_standard_matmul(a, b)
    }

    /// Compile Intel Xe optimized matrix multiplication
    ///
    /// Intel Arc GPUs feature Xe-Cores with XMX (Xe Matrix Extensions) for AI acceleration
    /// Similar to NVIDIA Tensor Cores and AMD Matrix Cores
    ///
    /// Note: Intel Xe GPU optimizations not yet fully implemented. Falls back to standard WGPU.
    /// Future implementation would use Intel oneAPI Level Zero or SYCL for XMX acceleration.
    fn compile_intel_xe_matmul<T: 'static>(
        &self,
        a: &Tensor<T>,
        b: &Tensor<T>,
        _xe_thread_groups: usize,
        _xmx_acceleration: bool,
        _intel_gpu_gen: &IntelGpuGeneration,
    ) -> Result<CompiledKernel> {
        // NOTE(v0.2): Implement Intel Xe GPU optimizations with XMX matrix extensions
        // For now, fallback to standard WGPU compute shader which works across all platforms
        self.compile_standard_matmul(a, b)
    }
}

// Pure, GPU-hardware-independent mapping helpers.
//
// Each function below is a total, side-effect-free mapping from already-queried real
// `wgpu` data (`AdapterInfo`/`Limits`/`Features`) to a `DeviceCapabilities` field. None
// of them touch a live device, so all of them are exhaustively unit-testable with
// literal `wgpu` values and require no GPU hardware to run in CI.

/// PCI vendor id for NVIDIA adapters, as reported by `wgpu::AdapterInfo::vendor`.
const PCI_VENDOR_NVIDIA: u32 = 0x10de;
/// PCI vendor id for AMD adapters.
const PCI_VENDOR_AMD: u32 = 0x1002;
/// PCI vendor id for Intel adapters.
const PCI_VENDOR_INTEL: u32 = 0x8086;
/// PCI vendor id for Apple adapters (Apple Silicon integrated GPU).
const PCI_VENDOR_APPLE: u32 = 0x106b;

/// Map a real `wgpu::AdapterInfo` to a [`GpuVendor`].
///
/// `AdapterInfo::vendor` is documented by wgpu as "generally...a 16-bit PCI vendor ID",
/// but this is backend-dependent: for example wgpu's Metal backend always reports
/// `vendor: 0` (it has no PCI bus to query), so on macOS the PCI id is never populated.
/// When the PCI vendor id doesn't match a known constant, this falls back to
/// case-insensitive substring matching against the adapter's real, self-reported name
/// (e.g. "Apple M2 Pro", "NVIDIA GeForce RTX 4090", "AMD Radeon RX 7900 XTX") before
/// giving up as [`GpuVendor::Unknown`]. Every branch is derived from real fields — there
/// is no fabricated default.
fn map_vendor(info: &wgpu::AdapterInfo) -> GpuVendor {
    match info.vendor {
        PCI_VENDOR_NVIDIA => GpuVendor::Nvidia,
        PCI_VENDOR_AMD => GpuVendor::AMD,
        PCI_VENDOR_INTEL => GpuVendor::Intel,
        PCI_VENDOR_APPLE => GpuVendor::Apple,
        _ => vendor_from_name(&info.name),
    }
}

/// Name-substring fallback used when the PCI vendor id is absent or unrecognized.
fn vendor_from_name(name: &str) -> GpuVendor {
    let lower = name.to_ascii_lowercase();
    if lower.contains("nvidia") || lower.contains("geforce") || lower.contains("quadro") {
        GpuVendor::Nvidia
    } else if lower.contains("amd") || lower.contains("radeon") {
        GpuVendor::AMD
    } else if lower.contains("intel") {
        GpuVendor::Intel
    } else if lower.contains("apple") {
        GpuVendor::Apple
    } else {
        GpuVendor::Unknown
    }
}

/// Derive an architecture/"compute capability" descriptor purely from real,
/// wgpu-reported adapter fields.
///
/// wgpu intentionally does not expose vendor-specific architecture numbering: there is
/// no API to obtain a CUDA "compute capability" (e.g. `8.9`), an AMD `gfx` ISA target
/// (e.g. `gfx1100`), or an Apple GPU family index. The previous implementation papered
/// over this gap with hardcoded per-vendor literals (`"7.5"` for *every* NVIDIA adapter,
/// `"gfx906"` for *every* AMD adapter, `"M1"` for *every* Apple adapter) — a fabricated
/// value for any device that wasn't that exact example part.
///
/// Instead, this reports the adapter's real, self-identified name together with its
/// backend (e.g. `"Apple M2 Pro (Metal)"`, `"NVIDIA GeForce RTX 4090 (Vulkan)"`), both
/// genuinely queried from `wgpu::AdapterInfo`. Callers should treat the result as an
/// opaque, human-readable descriptor rather than a parseable version number.
fn compute_capability_from_info(info: &wgpu::AdapterInfo) -> String {
    if info.name.is_empty() {
        format!("{:?}", info.backend)
    } else {
        format!("{} ({:?})", info.name, info.backend)
    }
}

/// Maximum compute-workgroup X size, read from a real `wgpu::Limits`.
///
/// This mirrors `DeviceProperties::max_threads_per_block` in
/// `crate::device::context::GpuContext::properties`, which uses the same
/// `max_compute_workgroup_size_x` field for the same purpose, so both real
/// "max threads per block" figures in this crate stay consistent with each other.
fn threads_from_limits(limits: &wgpu::Limits) -> u32 {
    limits.max_compute_workgroup_size_x
}

/// Maximum compute-workgroup storage (shared memory) size in bytes, read from a real
/// `wgpu::Limits`. Mirrors `DeviceProperties::shared_memory_per_block`.
fn shared_mem_from_limits(limits: &wgpu::Limits) -> u32 {
    limits.max_compute_workgroup_storage_size
}

/// fp16 shader support, read from a real `wgpu::Features`.
fn fp16_from_features(features: wgpu::Features) -> bool {
    features.contains(wgpu::Features::SHADER_F16)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Memory bandwidth cannot be measured without a vendor runtime / real
    /// device handle, so the query must return an honest error rather than the
    /// old fabricated literal `Ok(448.0)`.
    #[test]
    fn measure_memory_bandwidth_returns_honest_error_not_fabricated_literal() {
        let result = AdvancedKernelManager::measure_memory_bandwidth();
        assert!(
            result.is_err(),
            "measure_memory_bandwidth must not fabricate a measured value"
        );
        // Guard against regression back to the hardcoded 448.0 GB/s literal.
        assert_ne!(
            result.ok(),
            Some(448.0),
            "must not return the old fabricated 448.0 literal"
        );
    }

    /// Compute-unit count requires a real device handle; must err, not return
    /// the old fabricated `Ok(46)`.
    #[test]
    fn query_compute_units_returns_honest_error_not_fabricated_literal() {
        let result = AdvancedKernelManager::query_compute_units();
        assert!(result.is_err());
        assert_ne!(result.ok(), Some(46), "must not return the old 46 literal");
    }

    /// Tensor-core presence cannot be queried from WebGPU; must err, not the
    /// old fabricated `Ok(true)`.
    #[test]
    fn detect_tensor_cores_returns_honest_error() {
        assert!(AdvancedKernelManager::detect_tensor_cores().is_err());
    }

    /// Max threads per block now comes from a real `wgpu::Limits` value (via
    /// `threads_from_limits`), not the old fabricated `Ok(1024)`. Using a
    /// non-default, distinctive value proves this is a genuine field read rather
    /// than a hardcoded constant.
    #[test]
    fn query_max_threads_per_block_reflects_real_limits_not_fabricated_literal() {
        let limits = wgpu::Limits {
            max_compute_workgroup_size_x: 777,
            ..Default::default()
        };
        assert_eq!(
            AdvancedKernelManager::query_max_threads_per_block(&limits),
            777
        );
    }

    /// Shared-memory size now comes from a real `wgpu::Limits` value (via
    /// `shared_mem_from_limits`), not the old fabricated `Ok(49152)`.
    #[test]
    fn query_shared_memory_size_reflects_real_limits_not_fabricated_literal() {
        let limits = wgpu::Limits {
            max_compute_workgroup_storage_size: 12345,
            ..Default::default()
        };
        assert_eq!(
            AdvancedKernelManager::query_shared_memory_size(&limits),
            12345
        );
    }

    /// fp16 support now comes from a real `wgpu::Features` check (via
    /// `fp16_from_features`) rather than a blanket fabricated error.
    #[test]
    fn test_fp16_support_reflects_real_features_not_fabricated_error() {
        assert!(AdvancedKernelManager::test_fp16_support(
            wgpu::Features::SHADER_F16
        ));
        assert!(!AdvancedKernelManager::test_fp16_support(
            wgpu::Features::empty()
        ));
    }

    /// bf16 / cooperative-group capability still cannot be queried through wgpu at
    /// all (no equivalent feature flag exists on any backend); each must return an
    /// honest error rather than a fabricated boolean.
    #[test]
    fn bf16_and_coop_group_queries_return_honest_errors() {
        assert!(
            AdvancedKernelManager::test_bf16_support().is_err(),
            "bf16 support must not be fabricated"
        );
        assert!(
            AdvancedKernelManager::test_cooperative_groups().is_err(),
            "cooperative groups support must not be fabricated"
        );
    }

    /// Because `measure_memory_bandwidth`/`query_compute_units`/`detect_tensor_cores`/
    /// `test_bf16_support`/`test_cooperative_groups` still honestly error (wgpu has no
    /// API for any of them, on any backend, real adapter or not),
    /// `detect_device_capabilities` must propagate an error for a GPU device rather
    /// than returning fabricated capabilities -- even though several *other* fields
    /// (vendor, compute_capability, max_threads_per_block, shared_memory_size,
    /// supports_fp16) are now genuinely computed from real adapter data.
    #[cfg(feature = "gpu")]
    #[test]
    fn detect_device_capabilities_propagates_honest_error_for_gpu() {
        let result = AdvancedKernelManager::detect_device_capabilities(&Device::Gpu(0));
        assert!(
            result.is_err(),
            "GPU capability detection must not fabricate a DeviceCapabilities struct"
        );
    }

    // --- Pure mapping-helper tests -------------------------------------------------
    //
    // These construct literal `wgpu::AdapterInfo`/`Limits`/`Features` values by hand,
    // so they need no GPU and run identically in headless CI.

    /// Build a literal `wgpu::AdapterInfo` for pure-function tests. `AdapterInfo` is a
    /// plain data struct (no `Default` impl), so every field must be supplied.
    fn test_adapter_info(vendor: u32, name: &str, backend: wgpu::Backend) -> wgpu::AdapterInfo {
        wgpu::AdapterInfo {
            name: name.to_string(),
            vendor,
            device: 0,
            device_type: wgpu::DeviceType::Other,
            device_pci_bus_id: String::new(),
            driver: String::new(),
            driver_info: String::new(),
            backend,
            subgroup_min_size: 0,
            subgroup_max_size: 0,
            transient_saves_memory: Some(false),
            limit_bucket: None,
        }
    }

    #[test]
    fn map_vendor_detects_nvidia_by_pci_id() {
        let info = test_adapter_info(
            PCI_VENDOR_NVIDIA,
            "NVIDIA GeForce RTX 4090",
            wgpu::Backend::Vulkan,
        );
        assert_eq!(map_vendor(&info), GpuVendor::Nvidia);
    }

    #[test]
    fn map_vendor_detects_amd_by_pci_id() {
        let info = test_adapter_info(
            PCI_VENDOR_AMD,
            "AMD Radeon RX 7900 XTX",
            wgpu::Backend::Vulkan,
        );
        assert_eq!(map_vendor(&info), GpuVendor::AMD);
    }

    #[test]
    fn map_vendor_detects_intel_by_pci_id() {
        let info = test_adapter_info(PCI_VENDOR_INTEL, "Intel Arc A770", wgpu::Backend::Vulkan);
        assert_eq!(map_vendor(&info), GpuVendor::Intel);
    }

    #[test]
    fn map_vendor_detects_apple_by_pci_id() {
        let info = test_adapter_info(PCI_VENDOR_APPLE, "Apple GPU", wgpu::Backend::Metal);
        assert_eq!(map_vendor(&info), GpuVendor::Apple);
    }

    /// wgpu's Metal backend always reports `vendor: 0` (see wgpu-hal
    /// `src/metal/mod.rs`, `AdapterShared::expose`), so on macOS the name-substring
    /// fallback is what actually resolves vendor in practice -- this is exercised on
    /// every Metal-backed run of this crate, not just a theoretical corner case.
    #[test]
    fn map_vendor_falls_back_to_name_when_pci_id_is_zero_apple() {
        let info = test_adapter_info(0, "Apple M3 Max", wgpu::Backend::Metal);
        assert_eq!(map_vendor(&info), GpuVendor::Apple);
    }

    #[test]
    fn map_vendor_falls_back_to_name_for_nvidia_when_pci_id_unknown() {
        let info = test_adapter_info(0, "NVIDIA GeForce RTX 3080", wgpu::Backend::Vulkan);
        assert_eq!(map_vendor(&info), GpuVendor::Nvidia);
    }

    #[test]
    fn map_vendor_falls_back_to_name_for_amd_when_pci_id_unknown() {
        let info = test_adapter_info(0, "AMD Radeon Pro 5500M", wgpu::Backend::Vulkan);
        assert_eq!(map_vendor(&info), GpuVendor::AMD);
    }

    #[test]
    fn map_vendor_returns_unknown_for_unrecognized_adapter() {
        let info = test_adapter_info(0, "llvmpipe (LLVM 17.0.0, 256 bits)", wgpu::Backend::Vulkan);
        assert_eq!(map_vendor(&info), GpuVendor::Unknown);
    }

    #[test]
    fn threads_from_limits_reads_workgroup_size_x() {
        let limits_a = wgpu::Limits {
            max_compute_workgroup_size_x: 1024,
            ..Default::default()
        };
        assert_eq!(threads_from_limits(&limits_a), 1024);

        let limits_b = wgpu::Limits {
            max_compute_workgroup_size_x: 256,
            ..Default::default()
        };
        assert_eq!(
            threads_from_limits(&limits_b),
            256,
            "must track the real field, not a hardcoded constant"
        );
    }

    #[test]
    fn shared_mem_from_limits_reads_workgroup_storage_size() {
        let limits_a = wgpu::Limits {
            max_compute_workgroup_storage_size: 32768,
            ..Default::default()
        };
        assert_eq!(shared_mem_from_limits(&limits_a), 32768);

        let limits_b = wgpu::Limits {
            max_compute_workgroup_storage_size: 16384,
            ..Default::default()
        };
        assert_eq!(
            shared_mem_from_limits(&limits_b),
            16384,
            "must track the real field, not a hardcoded constant"
        );
    }

    #[test]
    fn fp16_from_features_true_when_shader_f16_present() {
        assert!(fp16_from_features(wgpu::Features::SHADER_F16));
    }

    #[test]
    fn fp16_from_features_false_when_absent() {
        assert!(!fp16_from_features(wgpu::Features::empty()));
    }

    #[test]
    fn compute_capability_from_info_uses_real_name_not_fabricated_version() {
        let info = test_adapter_info(
            PCI_VENDOR_NVIDIA,
            "NVIDIA GeForce RTX 4090",
            wgpu::Backend::Vulkan,
        );
        let cap = compute_capability_from_info(&info);
        assert!(
            cap.contains("RTX 4090"),
            "must reflect the real adapter name: {cap}"
        );
        // Guard against regression to the old fabricated "7.5" literal that every
        // NVIDIA adapter used to receive regardless of its actual model.
        assert_ne!(cap, "7.5");
    }

    #[test]
    fn compute_capability_from_info_differs_across_real_devices() {
        let a = compute_capability_from_info(&test_adapter_info(
            PCI_VENDOR_NVIDIA,
            "NVIDIA GeForce RTX 4090",
            wgpu::Backend::Vulkan,
        ));
        let b = compute_capability_from_info(&test_adapter_info(
            PCI_VENDOR_NVIDIA,
            "NVIDIA GeForce RTX 3050",
            wgpu::Backend::Vulkan,
        ));
        assert_ne!(
            a, b,
            "two different real NVIDIA cards must not collapse to the same fabricated literal"
        );
    }

    #[test]
    fn compute_capability_from_info_falls_back_to_backend_when_name_empty() {
        let info = test_adapter_info(0, "", wgpu::Backend::Vulkan);
        let cap = compute_capability_from_info(&info);
        assert!(!cap.is_empty());
    }

    #[test]
    fn detect_gpu_vendor_delegates_to_map_vendor() {
        let info = test_adapter_info(
            PCI_VENDOR_AMD,
            "AMD Radeon RX 7900 XTX",
            wgpu::Backend::Vulkan,
        );
        assert_eq!(
            AdvancedKernelManager::detect_gpu_vendor(&info),
            GpuVendor::AMD
        );
    }

    #[test]
    fn query_compute_capability_delegates_to_compute_capability_from_info() {
        let info = test_adapter_info(PCI_VENDOR_APPLE, "Apple M2 Pro", wgpu::Backend::Metal);
        let cap = AdvancedKernelManager::query_compute_capability(&info);
        assert!(cap.contains("M2 Pro"));
    }

    // --- Guarded real-adapter integration test --------------------------------------

    /// Local "is a real GPU adapter available" probe, matching the skip-gracefully
    /// convention already used in `tests/gpu_reduction_integration.rs`
    /// (`gpu_tests::gpu_available`).
    #[cfg(feature = "gpu")]
    fn gpu_adapter_available() -> bool {
        // Probe real device creation, not just adapter enumeration: an adapter
        // can be listed (e.g. via GL/EGL) in environments where `request_device`
        // then fails with "Parent device is lost". Gating on a genuinely usable
        // device makes these tests skip honestly instead of panicking.
        crate::gpu::gpu_device_available()
    }

    /// Integration test: when a real GPU adapter is available at test time, the real
    /// `wgpu::Adapter` -> `GpuAdapterCapabilities` -> pure-mapping-helper wiring must
    /// run against genuine hardware data without panicking, and must produce a
    /// non-empty, deterministic vendor/name mapping. `detect_device_capabilities`'s
    /// overall result is still `Err` by design (see
    /// `detect_device_capabilities_propagates_honest_error_for_gpu`), since several
    /// fields remain honest errors regardless of adapter availability -- this test
    /// exercises the real end-to-end wiring rather than only the pure helpers in
    /// isolation.
    ///
    /// Skips gracefully (does not fail the suite) when no GPU adapter is available.
    #[cfg(feature = "gpu")]
    #[test]
    fn detect_device_capabilities_uses_real_adapter_when_available() {
        if !gpu_adapter_available() {
            eprintln!(
                "GPU adapter not available, skipping detect_device_capabilities integration test"
            );
            return;
        }

        let caps = crate::device::get_gpu_adapter_capabilities(0).expect(
            "test: a GPU adapter was just reported available, so fetching its capabilities \
             must succeed",
        );
        assert!(
            !caps.info.name.is_empty(),
            "a real adapter must report a non-empty name"
        );

        let vendor = AdvancedKernelManager::detect_gpu_vendor(&caps.info);
        let vendor_again = AdvancedKernelManager::detect_gpu_vendor(&caps.info);
        assert_eq!(
            vendor, vendor_again,
            "vendor mapping must be a pure, deterministic function of real adapter info"
        );

        let result = AdvancedKernelManager::detect_device_capabilities(&Device::Gpu(0));
        assert!(
            result.is_err(),
            "capability detection must still honestly error on the fields wgpu cannot \
             answer, even with a real adapter available"
        );
    }

    // --- execute_metal_kernel now delegates to execute_wgpu_kernel ------------------

    /// `execute_metal_kernel` must now genuinely run the matmul (by delegating to
    /// the already-working `execute_wgpu_kernel`) instead of unconditionally
    /// returning the old "Metal Performance Shaders integration" error.
    ///
    /// This constructs an `AdvancedKernelManager` directly (bypassing the public
    /// `new()` constructor, which -- by design, see
    /// `detect_device_capabilities_propagates_honest_error_for_gpu` above --
    /// always errors for `Device::Gpu` today since several capability queries
    /// have no wgpu API at all) since `execute_metal_kernel` and everything it
    /// calls (`compile_standard_matmul`, `execute_wgpu_kernel`) only ever touch
    /// their `a`/`b`/`kernel` parameters, never `self`'s
    /// `device_info`/`kernel_cache`/`performance_data` -- so a manager built with
    /// placeholder capability data is a faithful test of the real dispatch path.
    #[cfg(feature = "metal")]
    #[test]
    fn execute_metal_kernel_delegates_to_wgpu_and_succeeds() {
        if !gpu_adapter_available() {
            eprintln!("GPU adapter not available, skipping execute_metal_kernel test");
            return;
        }

        let a_cpu = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
            .expect("test: create a");
        let b_cpu = Tensor::<f32>::from_vec(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2])
            .expect("test: create b");
        let a_gpu = a_cpu
            .to_device(Device::Gpu(0))
            .expect("test: move a to GPU");
        let b_gpu = b_cpu
            .to_device(Device::Gpu(0))
            .expect("test: move b to GPU");

        let manager = AdvancedKernelManager {
            device_info: Arc::new(RwLock::new(DeviceCapabilities {
                vendor: GpuVendor::Apple,
                compute_capability: "test".to_string(),
                memory_bandwidth: 0.0,
                compute_units: 0,
                has_tensor_cores: false,
                max_threads_per_block: 1024,
                shared_memory_size: 0,
                supports_fp16: false,
                supports_bf16: false,
                supports_coop_groups: false,
            })),
            kernel_cache: Arc::new(RwLock::new(HashMap::new())),
            performance_data: Arc::new(RwLock::new(HashMap::new())),
            strategy: KernelStrategy::StandardCompute,
        };
        // `execute_metal_kernel` ignores its `kernel` argument entirely (it builds
        // a fresh, correctly-shaped WGPU kernel internally via
        // `compile_standard_matmul`), so any well-formed placeholder value here
        // still exercises the real dispatch path.
        let dummy_kernel = CompiledKernel {
            id: "test".to_string(),
            strategy: KernelStrategy::StandardCompute,
            compiled_at: std::time::SystemTime::now(),
            parameters: KernelParameters {
                grid_size: (1, 1, 1),
                block_size: (1, 1, 1),
                shared_memory: 0,
                register_count: 0,
            },
            handle: KernelHandle::Metal {
                library: "test".to_string(),
                function: "test".to_string(),
            },
        };

        let result = manager
            .execute_metal_kernel(&a_gpu, &b_gpu, &dummy_kernel)
            .expect(
                "test: execute_metal_kernel must now succeed by delegating to execute_wgpu_kernel",
            );

        let result_cpu = result.to_cpu().expect("test: move result to CPU");
        assert_eq!(result_cpu.shape().dims(), &[1, 2, 2]);
        let data = result_cpu.data();
        let expected = [58.0f32, 64.0, 139.0, 154.0];
        for (got, want) in data.iter().zip(expected.iter()) {
            assert!(
                (got - want).abs() < 1e-4,
                "execute_metal_kernel result mismatch: got {got}, want {want}"
            );
        }
    }
}
