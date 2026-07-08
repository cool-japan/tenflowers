//! GPU Operations Module
//!
//! This module provides GPU acceleration for tensor operations using WGPU.
//! It includes optimized kernels for various operations and automatic device management.

use crate::gpu_profiler::global_profiler;
#[cfg(feature = "gpu")]
use crate::{buffer::TensorBuffer, Device, Result, TensorError};
use scirs2_core::ndarray::ArrayD;
use std::sync::Arc;
use std::time::Instant;

// Macro to safely include WGSL shader files, working around Rust 2021 edition prefix parsing
// This version is for use within this gpu.rs module only
macro_rules! include_shader {
    ("activation_ops") => {
        include_str!("../shaders/activation_ops.wgsl")
    };
    ("manipulation_ops") => {
        include_str!("../shaders/manipulation_ops.wgsl")
    };
    ("comparison_ops") => {
        include_str!("../shaders/comparison_ops.wgsl")
    };
    ("logical_ops") => {
        include_str!("../shaders/logical_ops.wgsl")
    };
    ("random_ops") => {
        include_str!("../shaders/random_ops.wgsl")
    };
    ("reduction_ops") => {
        include_str!("../shaders/reduction_ops.wgsl")
    };
    ("einsum_ops") => {
        include_str!("../shaders/einsum_ops.wgsl")
    };
    ("binary_ops") => {
        include_str!("../shaders/binary_ops.wgsl")
    };
    ("conv_ops") => {
        include_str!("../shaders/conv_ops.wgsl")
    };
    ("matmul_ops") => {
        include_str!("../shaders/matmul_ops.wgsl")
    };
    ("attention_ops") => {
        include_str!("../shaders/attention_ops.wgsl")
    };
    ("embedding_ops") => {
        include_str!("../shaders/embedding_ops.wgsl")
    };
    ("normalization_ops") => {
        include_str!("../shaders/normalization_ops.wgsl")
    };
    ("pooling_ops") => {
        include_str!("../shaders/pooling_ops.wgsl")
    };
    ("scan_ops") => {
        include_str!("../shaders/scan_ops.wgsl")
    };
    ("segmented_ops") => {
        include_str!("../shaders/segmented_ops.wgsl")
    };
    ("strided_ops") => {
        include_str!("../shaders/strided_ops.wgsl")
    };
    ("unary_ops") => {
        include_str!("../shaders/unary_ops.wgsl")
    };
    ("unary_ops_f64") => {
        include_str!("../shaders/unary_ops_f64.wgsl")
    };
    ("unary_ops_i32") => {
        include_str!("../shaders/unary_ops_i32.wgsl")
    };
    ("unary_ops_i64") => {
        include_str!("../shaders/unary_ops_i64.wgsl")
    };
    ("unary_ops_u32") => {
        include_str!("../shaders/unary_ops_u32.wgsl")
    };
    ("unary_ops_u64") => {
        include_str!("../shaders/unary_ops_u64.wgsl")
    };
    ("binary_ops_f64") => {
        include_str!("../shaders/binary_ops_f64.wgsl")
    };
    ("binary_ops_i32") => {
        include_str!("../shaders/binary_ops_i32.wgsl")
    };
    ("binary_ops_i64") => {
        include_str!("../shaders/binary_ops_i64.wgsl")
    };
    ("topk_ops") => {
        include_str!("../shaders/topk_ops.wgsl")
    };
    ("manipulation_ops2") => {
        include_str!("../shaders/manipulation_ops2.wgsl")
    };
    ("fused_ops") => {
        include_str!("../shaders/fused_ops.wgsl")
    };
    ("fft_ops") => {
        include_str!("../shaders/fft_ops.wgsl")
    };
}

/// GPU compute context for managing GPU resources
pub struct GpuContext {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
}

/// Binary scalar operation types for GPU kernels
pub enum BinaryScalarOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

impl GpuContext {
    /// Create a new GPU context
    pub fn new() -> Result<Self> {
        pollster::block_on(async {
            let instance =
                wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());

            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                    apply_limit_buckets: false,
                })
                .await
                .map_err(|_e| {
                    TensorError::gpu_error(
                        "GpuContext::new",
                        "Failed to find suitable GPU adapter",
                        None,
                        false,
                    )
                })?;

            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::empty(),
                    required_limits: if cfg!(target_arch = "wasm32") {
                        wgpu::Limits::downlevel_webgl2_defaults()
                    } else {
                        wgpu::Limits::default()
                    },
                    label: Some("TenfloweRS GPU Device"),
                    memory_hints: Default::default(),
                    experimental_features: wgpu::ExperimentalFeatures::default(),
                    trace: wgpu::Trace::default(),
                })
                .await
                .map_err(|e| {
                    TensorError::gpu_error(
                        "GpuContext::new",
                        &format!("Failed to create GPU device: {}", e),
                        None,
                        false,
                    )
                })?;

            Ok(Self {
                device: Arc::new(device),
                queue: Arc::new(queue),
            })
        })
    }

    /// Get or create the global GPU context
    pub fn global() -> Result<&'static Self> {
        use std::sync::OnceLock;
        static GLOBAL_CONTEXT: OnceLock<Result<GpuContext>> = OnceLock::new();

        GLOBAL_CONTEXT
            .get_or_init(|| GpuContext::new())
            .as_ref()
            .map_err(|e| e.clone())
    }
}

/// Runtime probe: can a *usable* GPU device actually be created right now?
///
/// This is deliberately stronger than `wgpu::Instance::request_adapter()`
/// succeeding. `request_adapter` returns an adapter whenever *any* ICD merely
/// enumerates one — including headless / sandboxed environments that list an
/// adapter (e.g. through GL/EGL) but have no working Vulkan loader, where the
/// subsequent `request_device` immediately fails with "Parent device is lost".
///
/// Callers and tests that need to run *real* GPU work (buffer upload, compute,
/// readback) must gate on this, not on adapter enumeration alone, so they skip
/// honestly when no device can be created instead of panicking on the first op.
///
/// The result is cached for the process lifetime via [`GpuContext::global`],
/// which performs the exact adapter + `request_device` creation the rest of the
/// crate's GPU paths use, so a `true` here means those paths will also succeed.
pub fn gpu_device_available() -> bool {
    GpuContext::global().is_ok()
}

/// Helper macro for including shaders - directly include to avoid scoping issues
/// Uses absolute paths from crate root to work from any calling context
#[macro_export]
macro_rules! gpu_include_shader {
    ("binary_ops") => {
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/gpu/shaders/binary_ops.wgsl"
        ))
    };
    ("binary_ops_f64") => {
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/gpu/shaders/binary_ops_f64.wgsl"
        ))
    };
    ("binary_ops_i32") => {
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/gpu/shaders/binary_ops_i32.wgsl"
        ))
    };
    ("binary_ops_i64") => {
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/gpu/shaders/binary_ops_i64.wgsl"
        ))
    };
    ("binary_ops_u32") => {
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/gpu/shaders/binary_ops_u32.wgsl"
        ))
    };
    ("binary_ops_u64") => {
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/gpu/shaders/binary_ops_u64.wgsl"
        ))
    };
    ("unary_ops") => {
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/gpu/shaders/unary_ops.wgsl"
        ))
    };
    ("unary_ops_f64") => {
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/gpu/shaders/unary_ops_f64.wgsl"
        ))
    };
    ("unary_ops_i32") => {
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/gpu/shaders/unary_ops_i32.wgsl"
        ))
    };
    ("unary_ops_i64") => {
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/gpu/shaders/unary_ops_i64.wgsl"
        ))
    };
    ("unary_ops_u32") => {
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/gpu/shaders/unary_ops_u32.wgsl"
        ))
    };
    ("unary_ops_u64") => {
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/gpu/shaders/unary_ops_u64.wgsl"
        ))
    };
    ("fft_ops") => {
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/gpu/shaders/fft_ops.wgsl"
        ))
    };
    ("einsum_ops") => {
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/gpu/shaders/einsum_ops.wgsl"
        ))
    };
    ("reduction_ops") => {
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/gpu/shaders/reduction_ops.wgsl"
        ))
    };
    ("matmul_ops") => {
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/gpu/shaders/matmul_ops.wgsl"
        ))
    };
    ("conv_ops") => {
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/gpu/shaders/conv_ops.wgsl"
        ))
    };
    ("attention_ops") => {
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/gpu/shaders/attention_ops.wgsl"
        ))
    };
    ("pooling_ops") => {
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/gpu/shaders/pooling_ops.wgsl"
        ))
    };
    ("activation_ops") => {
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/gpu/shaders/activation_ops.wgsl"
        ))
    };
    ("comparison_ops") => {
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/gpu/shaders/comparison_ops.wgsl"
        ))
    };
    ("logical_ops") => {
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/gpu/shaders/logical_ops.wgsl"
        ))
    };
    ("manipulation_ops") => {
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/gpu/shaders/manipulation_ops.wgsl"
        ))
    };
    ("manipulation_ops2") => {
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/gpu/shaders/manipulation_ops2.wgsl"
        ))
    };
    ("normalization_ops") => {
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/gpu/shaders/normalization_ops.wgsl"
        ))
    };
    ("embedding_ops") => {
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/gpu/shaders/embedding_ops.wgsl"
        ))
    };
    ("random_ops") => {
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/gpu/shaders/random_ops.wgsl"
        ))
    };
    ("scan_ops") => {
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/gpu/shaders/scan_ops.wgsl"
        ))
    };
    ("segmented_ops") => {
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/gpu/shaders/segmented_ops.wgsl"
        ))
    };
    ("strided_ops") => {
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/gpu/shaders/strided_ops.wgsl"
        ))
    };
    ("topk_ops") => {
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/gpu/shaders/topk_ops.wgsl"
        ))
    };
    ("fused_ops") => {
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/gpu/shaders/fused_ops.wgsl"
        ))
    };
}

pub use gpu_include_shader;

// Module declarations - organize functionality into logical groups

// Core GPU buffer and operation modules
pub mod binary_ops;
pub mod buffer;
pub mod logical_ops;
pub mod random_ops;
pub mod unary_ops;

// Async kernel execution module
#[cfg(feature = "gpu")]
pub mod async_kernel;

// Linear algebra operations module
#[cfg(feature = "gpu")]
pub mod linalg;

// Memory coalescing optimization module
#[cfg(feature = "gpu")]
pub mod memory_coalescing;

// Multi-stream GPU executor for CPU-GPU overlap
#[cfg(feature = "gpu")]
pub mod multi_stream_executor;

// RNN GPU operations module
#[cfg(feature = "gpu")]
pub mod rnn_ops;

// Attention operations module for neural networks
#[cfg(feature = "gpu")]
pub mod attention_ops;

// Kernel fusion module for performance optimization
#[cfg(feature = "gpu")]
pub mod kernel_fusion;

// Ultra-sophisticated fusion integration for production excellence
#[cfg(feature = "gpu")]
pub mod ultra_fusion_integration;

// Advanced memory pool management
#[cfg(feature = "gpu")]
pub mod memory_pool;

// GPU memory allocation tracing and diagnostics
#[cfg(feature = "gpu")]
pub mod memory_tracing;

// GPU memory diagnostics and profiling
#[cfg(feature = "gpu")]
pub mod memory_diagnostics;

// GPU reduction kernel templates and execution
#[cfg(feature = "gpu")]
pub mod reduction_kernels;

// Performance optimizer and profiler
#[cfg(feature = "gpu")]
pub mod performance_optimizer;

// Advanced kernel manager for cutting-edge GPU optimizations
#[cfg(feature = "gpu")]
pub mod advanced_kernel_manager;

// Platform-specific GPU backend modules
#[cfg(feature = "cudnn")]
pub mod cudnn;

#[cfg(all(target_os = "macos", feature = "metal"))]
pub mod metal_kernels;

#[cfg(feature = "rocm")]
pub mod rocm_kernels;

#[cfg(feature = "cuda")]
pub mod cuda_kernels;

#[cfg(feature = "nccl")]
pub mod nccl_integration;

// Modular GPU operations - NEW REFACTORED STRUCTURE
pub mod ops;

// Re-export commonly used types and functions
pub use binary_ops::{gpu_binary_op, BinaryOpKernel};
pub use buffer::{BufferManager, GpuBuffer, GpuBufferOps};
pub use unary_ops::{gpu_unary_op, UnaryOpKernel};

#[cfg(feature = "gpu")]
pub use attention_ops::*;
#[cfg(feature = "gpu")]
pub use kernel_fusion::*;
#[cfg(feature = "gpu")]
pub use linalg::*;
#[cfg(feature = "gpu")]
pub use ultra_fusion_integration::*;

// Re-export common types from ops module
pub use ops::ReductionOp;

// Re-export memory tracking functions
#[cfg(feature = "gpu")]
pub use memory_tracing::{
    current_gpu_memory_usage, generate_gpu_memory_report, peak_gpu_memory_usage,
    print_gpu_memory_report, MemoryReport, MemoryStats,
};

/// Trait for GPU operations on tensors
pub trait GpuOps {
    fn gpu_add(&self, other: &Self) -> crate::Result<Self>
    where
        Self: Sized;
    fn gpu_mul(&self, other: &Self) -> crate::Result<Self>
    where
        Self: Sized;
    fn gpu_sub(&self, other: &Self) -> crate::Result<Self>
    where
        Self: Sized;
    fn gpu_div(&self, other: &Self) -> crate::Result<Self>
    where
        Self: Sized;
}

/// Helper function to cast a scalar of a numeric type into `f32` for GPU shaders.
///
/// This dispatches on the concrete runtime type of `T` (via [`TypeId`](std::any::TypeId)) and performs
/// a value-preserving (lossless where the target permits) numeric conversion. For the
/// floating-point and integer primitives supported by the GPU backend this returns the
/// exact mathematical value cast to `f32`. For genuinely unknown types it returns
/// [`f32::NAN`] as an honest, loud signal rather than a plausible-but-fake constant.
fn cast_to_f32<T>(value: T) -> f32
where
    T: bytemuck::Pod + bytemuck::Zeroable + Clone + Send + Sync + 'static,
{
    use std::any::TypeId;

    let id = TypeId::of::<T>();

    if id == TypeId::of::<f32>() {
        // SAFETY: `id` equals `TypeId::of::<f32>()`, so `T` is exactly `f32` and the
        // reinterpret-read below is reading an `f32` from an `f32`.
        unsafe { *(&value as *const T as *const f32) }
    } else if id == TypeId::of::<f64>() {
        // SAFETY: `T` is `f64` (TypeId checked above).
        let v: f64 = unsafe { *(&value as *const T as *const f64) };
        v as f32
    } else if id == TypeId::of::<half::f16>() {
        // SAFETY: `T` is `half::f16` (TypeId checked above).
        let v: half::f16 = unsafe { *(&value as *const T as *const half::f16) };
        v.to_f32()
    } else if id == TypeId::of::<half::bf16>() {
        // SAFETY: `T` is `half::bf16` (TypeId checked above).
        let v: half::bf16 = unsafe { *(&value as *const T as *const half::bf16) };
        v.to_f32()
    } else if id == TypeId::of::<i8>() {
        // SAFETY: `T` is `i8` (TypeId checked above).
        let v: i8 = unsafe { *(&value as *const T as *const i8) };
        v as f32
    } else if id == TypeId::of::<u8>() {
        // SAFETY: `T` is `u8` (TypeId checked above).
        let v: u8 = unsafe { *(&value as *const T as *const u8) };
        v as f32
    } else if id == TypeId::of::<i16>() {
        // SAFETY: `T` is `i16` (TypeId checked above).
        let v: i16 = unsafe { *(&value as *const T as *const i16) };
        v as f32
    } else if id == TypeId::of::<u16>() {
        // SAFETY: `T` is `u16` (TypeId checked above).
        let v: u16 = unsafe { *(&value as *const T as *const u16) };
        v as f32
    } else if id == TypeId::of::<i32>() {
        // SAFETY: `T` is `i32` (TypeId checked above).
        let v: i32 = unsafe { *(&value as *const T as *const i32) };
        v as f32
    } else if id == TypeId::of::<u32>() {
        // SAFETY: `T` is `u32` (TypeId checked above).
        let v: u32 = unsafe { *(&value as *const T as *const u32) };
        v as f32
    } else if id == TypeId::of::<i64>() {
        // SAFETY: `T` is `i64` (TypeId checked above).
        let v: i64 = unsafe { *(&value as *const T as *const i64) };
        v as f32
    } else if id == TypeId::of::<u64>() {
        // SAFETY: `T` is `u64` (TypeId checked above).
        let v: u64 = unsafe { *(&value as *const T as *const u64) };
        v as f32
    } else {
        // Genuinely unknown numeric type: emit NaN so a downstream consumer sees a loud,
        // unmistakable signal instead of a silently fabricated value.
        f32::NAN
    }
}

/// GPU comparison operation dispatch function.
///
/// Computes the element-wise comparison of two GPU buffers and returns a
/// `GpuBuffer<u8>` where `0` represents `false` and `1` represents `true`.
///
/// The element type `T` is only constrained to be `bytemuck::Pod`, which does not
/// expose an ordering trait, so this function dispatches on the concrete runtime type
/// (via [`TypeId`](std::any::TypeId)) to perform a genuine per-element comparison with the correct
/// numeric semantics. The operands are read back from device memory, compared on the
/// host, and the boolean result is uploaded to a fresh device buffer. Unsupported
/// element types produce an honest [`TensorError`] rather than a fabricated all-true
/// result.
pub fn gpu_comparison_op_dispatch<T>(
    input_a: &GpuBuffer<T>,
    input_b: &GpuBuffer<T>,
    operation: self::ops::ComparisonOp,
) -> Result<GpuBuffer<u8>>
where
    T: bytemuck::Pod + bytemuck::Zeroable + Clone + Send + Sync + 'static,
{
    use std::any::TypeId;

    let device_id = match input_a.device_enum() {
        Device::Gpu(id) => id,
        _ => {
            return Err(TensorError::DeviceMismatch {
                operation: "comparison".to_string(),
                device1: format!("{:?}", input_a.device_enum()),
                device2: "GPU".to_string(),
                context: None,
            })
        }
    };

    let len = input_a.len();
    if input_b.len() != len {
        return Err(TensorError::invalid_argument(format!(
            "gpu_comparison_op_dispatch: operand lengths differ ({} vs {})",
            len,
            input_b.len()
        )));
    }

    // Read both operands back from device memory so the comparison operates on the
    // actual stored values rather than a constant.
    let host_a: Vec<T> = input_a.to_cpu()?;
    let host_b: Vec<T> = input_b.to_cpu()?;

    // Helper performing the per-element comparison for a concrete, ordered type.
    fn compare_elements<U: PartialOrd + PartialEq>(
        lhs: &[U],
        rhs: &[U],
        operation: self::ops::ComparisonOp,
    ) -> Vec<u8> {
        lhs.iter()
            .zip(rhs.iter())
            .map(|(l, r)| {
                let truthy = match operation {
                    self::ops::ComparisonOp::Eq => l == r,
                    self::ops::ComparisonOp::Ne => l != r,
                    self::ops::ComparisonOp::Lt => l < r,
                    self::ops::ComparisonOp::Le => l <= r,
                    self::ops::ComparisonOp::Gt => l > r,
                    self::ops::ComparisonOp::Ge => l >= r,
                };
                u8::from(truthy)
            })
            .collect()
    }

    // Reinterpret the host vectors as the concrete element type. The `TypeId` guard
    // guarantees the layouts are identical, so the slice view is sound.
    let id = TypeId::of::<T>();
    let result_data: Vec<u8> = if id == TypeId::of::<f32>() {
        // SAFETY: `T` is `f32`; the slices have identical layout and length.
        let lhs = unsafe { std::slice::from_raw_parts(host_a.as_ptr() as *const f32, len) };
        let rhs = unsafe { std::slice::from_raw_parts(host_b.as_ptr() as *const f32, len) };
        compare_elements(lhs, rhs, operation)
    } else if id == TypeId::of::<f64>() {
        // SAFETY: `T` is `f64`.
        let lhs = unsafe { std::slice::from_raw_parts(host_a.as_ptr() as *const f64, len) };
        let rhs = unsafe { std::slice::from_raw_parts(host_b.as_ptr() as *const f64, len) };
        compare_elements(lhs, rhs, operation)
    } else if id == TypeId::of::<i32>() {
        // SAFETY: `T` is `i32`.
        let lhs = unsafe { std::slice::from_raw_parts(host_a.as_ptr() as *const i32, len) };
        let rhs = unsafe { std::slice::from_raw_parts(host_b.as_ptr() as *const i32, len) };
        compare_elements(lhs, rhs, operation)
    } else if id == TypeId::of::<u32>() {
        // SAFETY: `T` is `u32`.
        let lhs = unsafe { std::slice::from_raw_parts(host_a.as_ptr() as *const u32, len) };
        let rhs = unsafe { std::slice::from_raw_parts(host_b.as_ptr() as *const u32, len) };
        compare_elements(lhs, rhs, operation)
    } else if id == TypeId::of::<i64>() {
        // SAFETY: `T` is `i64`.
        let lhs = unsafe { std::slice::from_raw_parts(host_a.as_ptr() as *const i64, len) };
        let rhs = unsafe { std::slice::from_raw_parts(host_b.as_ptr() as *const i64, len) };
        compare_elements(lhs, rhs, operation)
    } else if id == TypeId::of::<u64>() {
        // SAFETY: `T` is `u64`.
        let lhs = unsafe { std::slice::from_raw_parts(host_a.as_ptr() as *const u64, len) };
        let rhs = unsafe { std::slice::from_raw_parts(host_b.as_ptr() as *const u64, len) };
        compare_elements(lhs, rhs, operation)
    } else if id == TypeId::of::<i16>() {
        // SAFETY: `T` is `i16`.
        let lhs = unsafe { std::slice::from_raw_parts(host_a.as_ptr() as *const i16, len) };
        let rhs = unsafe { std::slice::from_raw_parts(host_b.as_ptr() as *const i16, len) };
        compare_elements(lhs, rhs, operation)
    } else if id == TypeId::of::<u16>() {
        // SAFETY: `T` is `u16`.
        let lhs = unsafe { std::slice::from_raw_parts(host_a.as_ptr() as *const u16, len) };
        let rhs = unsafe { std::slice::from_raw_parts(host_b.as_ptr() as *const u16, len) };
        compare_elements(lhs, rhs, operation)
    } else if id == TypeId::of::<i8>() {
        // SAFETY: `T` is `i8`.
        let lhs = unsafe { std::slice::from_raw_parts(host_a.as_ptr() as *const i8, len) };
        let rhs = unsafe { std::slice::from_raw_parts(host_b.as_ptr() as *const i8, len) };
        compare_elements(lhs, rhs, operation)
    } else if id == TypeId::of::<u8>() {
        // SAFETY: `T` is `u8`.
        let lhs = unsafe { std::slice::from_raw_parts(host_a.as_ptr() as *const u8, len) };
        let rhs = unsafe { std::slice::from_raw_parts(host_b.as_ptr() as *const u8, len) };
        compare_elements(lhs, rhs, operation)
    } else if id == TypeId::of::<half::f16>() {
        // SAFETY: `T` is `half::f16`.
        let lhs = unsafe { std::slice::from_raw_parts(host_a.as_ptr() as *const half::f16, len) };
        let rhs = unsafe { std::slice::from_raw_parts(host_b.as_ptr() as *const half::f16, len) };
        compare_elements(lhs, rhs, operation)
    } else if id == TypeId::of::<half::bf16>() {
        // SAFETY: `T` is `half::bf16`.
        let lhs = unsafe { std::slice::from_raw_parts(host_a.as_ptr() as *const half::bf16, len) };
        let rhs = unsafe { std::slice::from_raw_parts(host_b.as_ptr() as *const half::bf16, len) };
        compare_elements(lhs, rhs, operation)
    } else {
        return Err(TensorError::unsupported_operation_simple(format!(
            "gpu_comparison_op_dispatch does not support element type {}",
            std::any::type_name::<T>()
        )));
    };

    GpuBuffer::from_slice(&result_data, &Device::Gpu(device_id))
}

/// Execute an embedding lookup (row gather) operation on GPU buffers.
///
/// Gathers `total_indices` rows from the embedding table `weights`, shaped
/// `[num_embeddings, embedding_dim]`, using the integer positions stored in `indices`.
/// The result is a buffer of length `total_indices * embedding_dim` laid out as
/// `[total_indices, embedding_dim]`.
///
/// The operands are read back from device memory, the gather is computed on the host
/// with full bounds checking, and the gathered rows are uploaded to a fresh device
/// buffer. The index buffer is interpreted according to the concrete runtime type of
/// `T` (the supported integer types are `u32`, `i32`, `u64`, `i64`, `u16`, `i16`,
/// `u8`, `i8`); any other index type, an out-of-range index, or a buffer-size mismatch
/// yields an honest [`TensorError`] rather than a fabricated zero buffer.
pub fn execute_embedding_lookup<T>(
    indices: &GpuBuffer<T>,
    weights: &GpuBuffer<T>,
    num_embeddings: usize,
    embedding_dim: usize,
    total_indices: usize,
) -> Result<GpuBuffer<T>>
where
    T: bytemuck::Pod + bytemuck::Zeroable + Clone + Send + Sync + 'static + Default,
{
    use std::any::TypeId;

    let output_size = total_indices * embedding_dim;

    // Get device from indices buffer.
    let device_id = match indices.device_enum() {
        Device::Gpu(id) => id,
        _ => {
            return Err(TensorError::DeviceMismatch {
                operation: "embedding_lookup".to_string(),
                device1: format!("{:?}", indices.device_enum()),
                device2: "GPU".to_string(),
                context: None,
            })
        }
    };

    // Validate the declared geometry against the actual buffer lengths.
    if indices.len() < total_indices {
        return Err(TensorError::invalid_argument(format!(
            "execute_embedding_lookup: index buffer holds {} elements but {} were requested",
            indices.len(),
            total_indices
        )));
    }
    let expected_weight_len = num_embeddings * embedding_dim;
    if weights.len() != expected_weight_len {
        return Err(TensorError::invalid_argument(format!(
            "execute_embedding_lookup: weight buffer holds {} elements, expected {} ({} x {})",
            weights.len(),
            expected_weight_len,
            num_embeddings,
            embedding_dim
        )));
    }

    // Read both operands back from device memory.
    let host_indices: Vec<T> = indices.to_cpu()?;
    let host_weights: Vec<T> = weights.to_cpu()?;

    // Decode the index buffer into host `usize` positions, dispatching on the concrete
    // runtime element type. Returns an error for unsupported (non-integer) index types.
    let positions: Vec<usize> = {
        let id = TypeId::of::<T>();
        let decode = |raw: i128| -> Result<usize> {
            if raw < 0 {
                return Err(TensorError::invalid_argument(format!(
                    "execute_embedding_lookup: negative embedding index {}",
                    raw
                )));
            }
            usize::try_from(raw).map_err(|_| {
                TensorError::invalid_argument(format!(
                    "execute_embedding_lookup: embedding index {} does not fit in usize",
                    raw
                ))
            })
        };

        if id == TypeId::of::<u32>() {
            // SAFETY: `T` is `u32`.
            let raw = unsafe {
                std::slice::from_raw_parts(host_indices.as_ptr() as *const u32, total_indices)
            };
            raw.iter()
                .map(|&v| decode(v as i128))
                .collect::<Result<_>>()?
        } else if id == TypeId::of::<i32>() {
            // SAFETY: `T` is `i32`.
            let raw = unsafe {
                std::slice::from_raw_parts(host_indices.as_ptr() as *const i32, total_indices)
            };
            raw.iter()
                .map(|&v| decode(v as i128))
                .collect::<Result<_>>()?
        } else if id == TypeId::of::<u64>() {
            // SAFETY: `T` is `u64`.
            let raw = unsafe {
                std::slice::from_raw_parts(host_indices.as_ptr() as *const u64, total_indices)
            };
            raw.iter()
                .map(|&v| decode(v as i128))
                .collect::<Result<_>>()?
        } else if id == TypeId::of::<i64>() {
            // SAFETY: `T` is `i64`.
            let raw = unsafe {
                std::slice::from_raw_parts(host_indices.as_ptr() as *const i64, total_indices)
            };
            raw.iter()
                .map(|&v| decode(v as i128))
                .collect::<Result<_>>()?
        } else if id == TypeId::of::<u16>() {
            // SAFETY: `T` is `u16`.
            let raw = unsafe {
                std::slice::from_raw_parts(host_indices.as_ptr() as *const u16, total_indices)
            };
            raw.iter()
                .map(|&v| decode(v as i128))
                .collect::<Result<_>>()?
        } else if id == TypeId::of::<i16>() {
            // SAFETY: `T` is `i16`.
            let raw = unsafe {
                std::slice::from_raw_parts(host_indices.as_ptr() as *const i16, total_indices)
            };
            raw.iter()
                .map(|&v| decode(v as i128))
                .collect::<Result<_>>()?
        } else if id == TypeId::of::<u8>() {
            // SAFETY: `T` is `u8`.
            let raw = unsafe {
                std::slice::from_raw_parts(host_indices.as_ptr() as *const u8, total_indices)
            };
            raw.iter()
                .map(|&v| decode(v as i128))
                .collect::<Result<_>>()?
        } else if id == TypeId::of::<i8>() {
            // SAFETY: `T` is `i8`.
            let raw = unsafe {
                std::slice::from_raw_parts(host_indices.as_ptr() as *const i8, total_indices)
            };
            raw.iter()
                .map(|&v| decode(v as i128))
                .collect::<Result<_>>()?
        } else {
            return Err(TensorError::unsupported_operation_simple(format!(
                "execute_embedding_lookup expects an integer index type, got {}",
                std::any::type_name::<T>()
            )));
        }
    };

    // Gather the embedding rows. Each position selects a contiguous `embedding_dim`
    // slice of the weight table; out-of-range positions are a hard error.
    let mut result_data: Vec<T> = Vec::with_capacity(output_size);
    for &position in &positions {
        if position >= num_embeddings {
            return Err(TensorError::invalid_argument(format!(
                "execute_embedding_lookup: index {} out of range for {} embeddings",
                position, num_embeddings
            )));
        }
        let start = position * embedding_dim;
        result_data.extend_from_slice(&host_weights[start..start + embedding_dim]);
    }

    GpuBuffer::from_slice(&result_data, &Device::Gpu(device_id))
}
