//! WASM type definitions: structs and data types for the WebAssembly platform support.

#[cfg(target_arch = "wasm32")]
use js_sys::WebAssembly;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use web_sys::{console, window, Performance};

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use std::arch::wasm32::*;

#[allow(unused_imports)]
use crate::{DType, Device, Result, TensorError};
#[allow(unused_imports)]
use std::collections::HashMap;

#[cfg(target_arch = "wasm32")]
#[allow(unused_imports)]
use super::memory::WasmAllocator;

/// WebAssembly-specific tensor operations
#[cfg(target_arch = "wasm32")]
pub struct WasmTensorOps {
    /// WebAssembly memory instance
    pub(crate) memory: Option<WebAssembly::Memory>,
    /// Performance interface for timing
    pub(crate) performance: Option<Performance>,
}

#[cfg(target_arch = "wasm32")]
impl Default for WasmTensorOps {
    fn default() -> Self {
        Self::new()
    }
}

/// WebAssembly device context
#[cfg(target_arch = "wasm32")]
pub struct WasmContext {
    pub(crate) ops: WasmTensorOps,
    pub(crate) memory_limit: usize,
}

#[cfg(target_arch = "wasm32")]
impl Default for WasmContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Performance timer for WASM
#[cfg(target_arch = "wasm32")]
pub struct WasmTimer {
    pub(crate) performance: Option<Performance>,
    pub(crate) start_time: f64,
}

/// WASM feature detection
#[cfg(target_arch = "wasm32")]
pub struct WasmFeatures {
    pub simd: bool,
    pub threads: bool,
    pub bulk_memory: bool,
    pub reference_types: bool,
}

/// WebGPU device limits
#[cfg(target_arch = "wasm32")]
#[derive(Debug, Clone)]
pub struct WebGpuLimits {
    pub max_texture_dimension_1d: u32,
    pub max_texture_dimension_2d: u32,
    pub max_texture_dimension_3d: u32,
    pub max_bind_groups: u32,
    pub max_storage_buffer_binding_size: usize,
    pub max_compute_workgroup_size_x: u32,
    pub max_compute_workgroup_size_y: u32,
    pub max_compute_workgroup_size_z: u32,
    pub max_compute_workgroups_per_dimension: u32,
}

/// WebGPU-based tensor operations for WASM
#[cfg(target_arch = "wasm32")]
pub struct WebGpuBackend {
    pub(crate) device: Option<web_sys::GpuDevice>,
    pub(crate) queue: Option<web_sys::GpuQueue>,
    pub(crate) adapter: Option<web_sys::GpuAdapter>,
    pub(crate) supported_features: Option<web_sys::GpuSupportedFeatures>,
    pub(crate) limits: Option<web_sys::GpuSupportedLimits>,
    pub(crate) shader_cache: std::cell::RefCell<HashMap<String, web_sys::GpuShaderModule>>,
    pub(crate) compute_pipeline_cache:
        std::cell::RefCell<HashMap<String, web_sys::GpuComputePipeline>>,
}

/// Enhanced WASM context with WebGPU support
#[cfg(target_arch = "wasm32")]
pub struct WasmContextWithGpu {
    pub(crate) cpu_ops: WasmTensorOps,
    pub(crate) gpu_backend: Option<WebGpuBackend>,
    pub(crate) prefer_gpu: bool,
    pub(crate) gpu_threshold: usize, // Minimum size to use GPU
}

/// JavaScript-exposed WebGPU context wrapper
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct WasmWebGpuContext {
    pub(crate) inner: WasmContextWithGpu,
}

// Non-WASM fallback context
#[cfg(not(target_arch = "wasm32"))]
pub struct WasmContext;

#[cfg(not(target_arch = "wasm32"))]
impl WasmContext {
    pub fn new() -> Self {
        Self
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Default for WasmContext {
    fn default() -> Self {
        Self::new()
    }
}
