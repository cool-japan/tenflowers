//! WebAssembly platform support for TenfloweRS
//!
//! This module provides WebAssembly-specific implementations and optimizations
//! for running TenfloweRS in web browsers and WASI environments.
//!
//! Features:
//! - Basic WASM tensor operations
//! - SIMD-optimized operations when available
//! - WebGPU backend for GPU acceleration in browsers
//! - Memory management for WASM constraints
//! - Performance monitoring and timing

pub mod bindings;
pub mod memory;
pub mod ops;
pub mod tests;
pub mod types;

// Re-export all public items so callers continue to access them via `crate::wasm::*`.
// Items that only exist on wasm32 targets are guarded with the appropriate cfg attribute.
#[cfg(target_arch = "wasm32")]
pub use memory::WasmAllocator;
#[cfg(target_arch = "wasm32")]
pub use ops::WasmOpRegistry;
#[cfg(target_arch = "wasm32")]
pub use types::{
    WasmContext, WasmContextWithGpu, WasmFeatures, WasmTensorOps, WasmTimer, WasmWebGpuContext,
    WebGpuBackend, WebGpuLimits,
};
// On non-wasm32 targets only the stub WasmContext is available.
#[cfg(not(target_arch = "wasm32"))]
pub use types::WasmContext;

/// Utility functions for WASM integration
pub mod utils {
    /// Check if running in WASM environment
    pub fn is_wasm() -> bool {
        cfg!(target_arch = "wasm32")
    }

    /// Get optimal chunk size for WASM operations
    pub fn optimal_chunk_size() -> usize {
        if is_wasm() {
            // Smaller chunks for WASM to avoid stack overflow
            1024
        } else {
            // Larger chunks for native
            8192
        }
    }

    /// Get recommended memory limit for WASM
    pub fn recommended_memory_limit() -> usize {
        if is_wasm() {
            256 * 1024 * 1024 // 256MB for WASM
        } else {
            2 * 1024 * 1024 * 1024 // 2GB for native
        }
    }
}
