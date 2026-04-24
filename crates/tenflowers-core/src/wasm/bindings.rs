//! WASM JavaScript bindings: `#[wasm_bindgen]` exported functions and the WasmWebGpuContext impl.

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use web_sys::console;

#[cfg(target_arch = "wasm32")]
use super::types::{WasmContextWithGpu, WasmFeatures, WasmWebGpuContext, WebGpuBackend};

// Export WASM bindings
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn wasm_add_arrays(a: &[f32], b: &[f32]) -> Vec<f32> {
    if a.len() != b.len() {
        console::log_1(&"Array lengths must match".into());
        return vec![];
    }

    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn wasm_tensor_info() -> String {
    let features = WasmFeatures::detect();
    let compile_time_simd = cfg!(all(target_arch = "wasm32", target_feature = "simd128"));
    let webgpu_available = WebGpuBackend::is_available();

    format!(
        "TenfloweRS WASM Support v0.1.1 - Features: SIMD: {} (compile-time: {}), Threads: {}, WebGPU: {}, Bulk Memory: {}, Reference Types: {}, Operations: add, mul, sub, relu with SIMD/WebGPU optimization",
        features.simd, compile_time_simd, features.threads, webgpu_available, features.bulk_memory, features.reference_types
    )
}

/// Check WebGPU availability from JavaScript
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn wasm_webgpu_available() -> bool {
    WebGpuBackend::is_available()
}

/// WasmWebGpuContext JavaScript bindings
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl WasmWebGpuContext {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: WasmContextWithGpu::new(),
        }
    }

    /// Initialize WebGPU backend (must be called from JavaScript async context)
    #[wasm_bindgen]
    pub async fn init_gpu(&mut self) -> bool {
        self.inner.init_gpu().await.is_ok()
    }

    /// Get GPU information if available
    #[wasm_bindgen]
    pub fn gpu_info(&self) -> Option<String> {
        self.inner.gpu_info()
    }

    /// Set the threshold for GPU usage (number of elements)
    #[wasm_bindgen]
    pub fn set_gpu_threshold(&mut self, threshold: usize) {
        self.inner.set_gpu_threshold(threshold);
    }

    /// Enable or disable GPU preference
    #[wasm_bindgen]
    pub fn set_prefer_gpu(&mut self, prefer: bool) {
        self.inner.set_prefer_gpu(prefer);
    }
}
