//! WASM operation implementations: tensor ops, GPU backend, feature detection, and op registry.

#[cfg(target_arch = "wasm32")]
use js_sys::{ArrayBuffer, Float32Array, Uint8Array, WebAssembly};
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
use super::types::{
    WasmContext, WasmContextWithGpu, WasmFeatures, WasmTensorOps, WasmTimer, WebGpuBackend,
    WebGpuLimits,
};

// ================================
// WasmTensorOps implementation
// ================================

#[cfg(target_arch = "wasm32")]
impl WasmTensorOps {
    /// Create a new WASM tensor operations instance
    pub fn new() -> Self {
        let performance = window().and_then(|win| win.performance());

        Self {
            memory: None,
            performance,
        }
    }

    /// Initialize WebAssembly memory
    pub fn init_memory(&mut self, initial_pages: u32) -> Result<()> {
        let memory_descriptor = js_sys::Object::new();
        js_sys::Reflect::set(&memory_descriptor, &"initial".into(), &initial_pages.into())
            .map_err(|_| TensorError::device_error_simple("Failed to set memory descriptor"))?;

        let memory = WebAssembly::Memory::new(&memory_descriptor)
            .map_err(|_| TensorError::device_error_simple("Failed to create WASM memory"))?;

        self.memory = Some(memory);
        Ok(())
    }

    /// Get current time for performance measurement
    pub fn now(&self) -> f64 {
        self.performance.as_ref().map(|p| p.now()).unwrap_or(0.0)
    }

    /// Log message to browser console
    pub fn log(&self, message: &str) {
        console::log_1(&message.into());
    }

    /// Create a typed array from tensor data
    pub fn create_float32_array(&self, data: &[f32]) -> Result<Float32Array> {
        let array = Float32Array::new_with_length(data.len() as u32);
        for (i, &value) in data.iter().enumerate() {
            array.set_index(i as u32, value);
        }
        Ok(array)
    }

    /// Create tensor data from typed array
    pub fn from_float32_array(&self, array: &Float32Array) -> Result<Vec<f32>> {
        let length = array.length() as usize;
        let mut data = Vec::with_capacity(length);

        for i in 0..length {
            data.push(array.get_index(i as u32));
        }

        Ok(data)
    }

    /// Perform basic arithmetic operations using WASM SIMD if available
    pub fn add_simd(&self, a: &[f32], b: &[f32], result: &mut [f32]) -> Result<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return Err(TensorError::invalid_shape_simple(
                "Array lengths must match",
            ));
        }

        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            self.add_simd_optimized(a, b, result)
        }
        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
        {
            self.add_scalar(a, b, result)
        }
    }

    /// SIMD-optimized addition for WASM
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    fn add_simd_optimized(&self, a: &[f32], b: &[f32], result: &mut [f32]) -> Result<()> {
        let len = a.len();
        let simd_len = (len / 4) * 4; // Process 4 elements at a time

        // SIMD processing for chunks of 4
        for i in (0..simd_len).step_by(4) {
            unsafe {
                // Load 4 f32 values into SIMD registers
                let a_vec = v128_load(a.as_ptr().add(i) as *const v128);
                let b_vec = v128_load(b.as_ptr().add(i) as *const v128);

                // Perform SIMD addition
                let result_vec = f32x4_add(a_vec, b_vec);

                // Store result
                v128_store(result.as_mut_ptr().add(i) as *mut v128, result_vec);
            }
        }

        // Handle remaining elements with scalar operations
        for i in simd_len..len {
            result[i] = a[i] + b[i];
        }

        Ok(())
    }

    /// Scalar fallback for addition
    fn add_scalar(&self, a: &[f32], b: &[f32], result: &mut [f32]) -> Result<()> {
        for (i, (a_val, b_val)) in a.iter().zip(b.iter()).enumerate() {
            result[i] = a_val + b_val;
        }
        Ok(())
    }

    /// SIMD-optimized multiplication
    pub fn mul_simd(&self, a: &[f32], b: &[f32], result: &mut [f32]) -> Result<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return Err(TensorError::invalid_shape_simple(
                "Array lengths must match",
            ));
        }

        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            self.mul_simd_optimized(a, b, result)
        }
        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
        {
            self.mul_scalar(a, b, result)
        }
    }

    /// SIMD-optimized multiplication for WASM
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    fn mul_simd_optimized(&self, a: &[f32], b: &[f32], result: &mut [f32]) -> Result<()> {
        let len = a.len();
        let simd_len = (len / 4) * 4;

        for i in (0..simd_len).step_by(4) {
            unsafe {
                let a_vec = v128_load(a.as_ptr().add(i) as *const v128);
                let b_vec = v128_load(b.as_ptr().add(i) as *const v128);
                let result_vec = f32x4_mul(a_vec, b_vec);
                v128_store(result.as_mut_ptr().add(i) as *mut v128, result_vec);
            }
        }

        // Handle remaining elements
        for i in simd_len..len {
            result[i] = a[i] * b[i];
        }

        Ok(())
    }

    /// Scalar fallback for multiplication
    fn mul_scalar(&self, a: &[f32], b: &[f32], result: &mut [f32]) -> Result<()> {
        for (i, (a_val, b_val)) in a.iter().zip(b.iter()).enumerate() {
            result[i] = a_val * b_val;
        }
        Ok(())
    }

    /// SIMD-optimized subtraction
    pub fn sub_simd(&self, a: &[f32], b: &[f32], result: &mut [f32]) -> Result<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return Err(TensorError::invalid_shape_simple(
                "Array lengths must match",
            ));
        }

        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            self.sub_simd_optimized(a, b, result)
        }
        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
        {
            self.sub_scalar(a, b, result)
        }
    }

    /// SIMD-optimized subtraction for WASM
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    fn sub_simd_optimized(&self, a: &[f32], b: &[f32], result: &mut [f32]) -> Result<()> {
        let len = a.len();
        let simd_len = (len / 4) * 4;

        for i in (0..simd_len).step_by(4) {
            unsafe {
                let a_vec = v128_load(a.as_ptr().add(i) as *const v128);
                let b_vec = v128_load(b.as_ptr().add(i) as *const v128);
                let result_vec = f32x4_sub(a_vec, b_vec);
                v128_store(result.as_mut_ptr().add(i) as *mut v128, result_vec);
            }
        }

        for i in simd_len..len {
            result[i] = a[i] - b[i];
        }

        Ok(())
    }

    /// Scalar fallback for subtraction
    fn sub_scalar(&self, a: &[f32], b: &[f32], result: &mut [f32]) -> Result<()> {
        for (i, (a_val, b_val)) in a.iter().zip(b.iter()).enumerate() {
            result[i] = a_val - b_val;
        }
        Ok(())
    }

    /// SIMD-optimized activation functions
    pub fn relu_simd(&self, input: &[f32], result: &mut [f32]) -> Result<()> {
        if input.len() != result.len() {
            return Err(TensorError::invalid_shape_simple(
                "Array lengths must match",
            ));
        }

        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            self.relu_simd_optimized(input, result)
        }
        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
        {
            self.relu_scalar(input, result)
        }
    }

    /// SIMD-optimized ReLU
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    fn relu_simd_optimized(&self, input: &[f32], result: &mut [f32]) -> Result<()> {
        let len = input.len();
        let simd_len = (len / 4) * 4;
        let zero_vec = f32x4_splat(0.0);

        for i in (0..simd_len).step_by(4) {
            unsafe {
                let input_vec = v128_load(input.as_ptr().add(i) as *const v128);
                let result_vec = f32x4_max(input_vec, zero_vec);
                v128_store(result.as_mut_ptr().add(i) as *mut v128, result_vec);
            }
        }

        for i in simd_len..len {
            result[i] = input[i].max(0.0);
        }

        Ok(())
    }

    /// Scalar fallback for ReLU
    fn relu_scalar(&self, input: &[f32], result: &mut [f32]) -> Result<()> {
        for (i, &val) in input.iter().enumerate() {
            result[i] = val.max(0.0);
        }
        Ok(())
    }

    /// Perform matrix multiplication optimized for WASM
    pub fn matmul_wasm(
        &self,
        a: &[f32],
        b: &[f32],
        result: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        if a.len() != m * k || b.len() != k * n || result.len() != m * n {
            return Err(TensorError::invalid_shape_simple(
                "Matrix dimensions don't match",
            ));
        }

        // Cache-friendly matrix multiplication
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += a[i * k + l] * b[l * n + j];
                }
                result[i * n + j] = sum;
            }
        }

        Ok(())
    }
}

// ================================
// WasmContext implementation
// ================================

#[cfg(target_arch = "wasm32")]
impl WasmContext {
    /// Create a new WASM context
    pub fn new() -> Self {
        Self {
            ops: WasmTensorOps::new(),
            memory_limit: 256 * 1024 * 1024, // 256MB default
        }
    }

    /// Create a WASM context with custom memory limit
    pub fn with_memory_limit(memory_limit: usize) -> Self {
        Self {
            ops: WasmTensorOps::new(),
            memory_limit,
        }
    }

    /// Get the tensor operations
    pub fn ops(&self) -> &WasmTensorOps {
        &self.ops
    }

    /// Get mutable tensor operations
    pub fn ops_mut(&mut self) -> &mut WasmTensorOps {
        &mut self.ops
    }

    /// Check available memory
    pub fn available_memory(&self) -> usize {
        // In WASM, we use our configured limit
        self.memory_limit
    }

    /// Create a performance timer
    pub fn create_timer(&self) -> WasmTimer {
        WasmTimer::new(self.ops.performance.clone())
    }
}

// ================================
// WasmTimer implementation
// ================================

#[cfg(target_arch = "wasm32")]
impl WasmTimer {
    pub fn new(performance: Option<Performance>) -> Self {
        let start_time = performance.as_ref().map(|p| p.now()).unwrap_or(0.0);

        Self {
            performance,
            start_time,
        }
    }

    /// Get elapsed time in milliseconds
    pub fn elapsed(&self) -> f64 {
        let current_time = self.performance.as_ref().map(|p| p.now()).unwrap_or(0.0);

        current_time - self.start_time
    }
}

// ================================
// WasmFeatures implementation
// ================================

#[cfg(target_arch = "wasm32")]
impl WasmFeatures {
    /// Detect available WASM features
    pub fn detect() -> Self {
        let simd = Self::detect_simd();
        let threads = Self::detect_threads();

        Self {
            simd,
            threads,
            bulk_memory: true,     // Generally available in modern browsers
            reference_types: true, // Generally available in modern browsers
        }
    }

    /// Detect WASM SIMD support
    fn detect_simd() -> bool {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            // If compiled with simd128 feature, SIMD is available
            true
        }
        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
        {
            // Runtime detection could be added here using JavaScript
            // For now, conservatively assume no SIMD
            false
        }
    }

    /// Detect WASM threads support (SharedArrayBuffer)
    fn detect_threads() -> bool {
        // This would require checking for SharedArrayBuffer availability
        // and proper COOP/COEP headers in the browser
        #[cfg(target_arch = "wasm32")]
        {
            // For now, conservatively assume no threads
            false
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            false
        }
    }

    /// Check if SIMD operations are available
    pub fn has_simd(&self) -> bool {
        self.simd
    }

    /// Check if multi-threading is available
    pub fn has_threads(&self) -> bool {
        self.threads
    }
}

// ================================
// WasmOpRegistry
// ================================

/// WASM-specific tensor operations registry
#[cfg(target_arch = "wasm32")]
pub struct WasmOpRegistry {
    operations: HashMap<String, Box<dyn Fn(&[f32], &[f32]) -> Result<Vec<f32>> + Send + Sync>>,
}

#[cfg(target_arch = "wasm32")]
impl WasmOpRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            operations: HashMap::new(),
        };

        // Register basic operations
        registry.register_basic_ops();
        registry
    }

    fn register_basic_ops(&mut self) {
        let ops = WasmTensorOps::new();

        // Add operation with SIMD optimization
        self.operations.insert(
            "add".to_string(),
            Box::new(move |a: &[f32], b: &[f32]| -> Result<Vec<f32>> {
                if a.len() != b.len() {
                    return Err(TensorError::invalid_shape_simple(
                        "Arrays must have same length",
                    ));
                }
                let mut result = vec![0.0; a.len()];
                ops.add_simd(a, b, &mut result)?;
                Ok(result)
            }),
        );

        // Multiply operation with SIMD optimization
        self.operations.insert(
            "mul".to_string(),
            Box::new(move |a: &[f32], b: &[f32]| -> Result<Vec<f32>> {
                if a.len() != b.len() {
                    return Err(TensorError::invalid_shape_simple(
                        "Arrays must have same length",
                    ));
                }
                let mut result = vec![0.0; a.len()];
                ops.mul_simd(a, b, &mut result)?;
                Ok(result)
            }),
        );

        // Subtract operation with SIMD optimization
        self.operations.insert(
            "sub".to_string(),
            Box::new(move |a: &[f32], b: &[f32]| -> Result<Vec<f32>> {
                if a.len() != b.len() {
                    return Err(TensorError::invalid_shape_simple(
                        "Arrays must have same length",
                    ));
                }
                let mut result = vec![0.0; a.len()];
                ops.sub_simd(a, b, &mut result)?;
                Ok(result)
            }),
        );
    }

    pub fn execute(&self, op_name: &str, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        if let Some(op) = self.operations.get(op_name) {
            op(a, b)
        } else {
            Err(TensorError::unsupported_operation_simple(&format!(
                "Operation '{}' not supported in WASM",
                op_name
            )))
        }
    }
}

#[cfg(target_arch = "wasm32")]
impl Default for WasmOpRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ================================
// WebGpuBackend implementation
// ================================

#[cfg(target_arch = "wasm32")]
impl WebGpuBackend {
    /// Create a new WebGPU backend
    pub fn new() -> Self {
        Self {
            device: None,
            queue: None,
            adapter: None,
            supported_features: None,
            limits: None,
            shader_cache: std::cell::RefCell::new(HashMap::new()),
            compute_pipeline_cache: std::cell::RefCell::new(HashMap::new()),
        }
    }

    /// Initialize WebGPU backend
    pub async fn initialize(&mut self) -> Result<()> {
        let window = web_sys::window()
            .ok_or_else(|| TensorError::device_error_simple("No window object available"))?;

        let navigator = window.navigator();
        let gpu = navigator
            .gpu()
            .ok_or_else(|| TensorError::device_error_simple("WebGPU not supported"))?;

        // Request adapter
        let adapter_options = web_sys::GpuRequestAdapterOptions::new();
        adapter_options.set_power_preference(web_sys::GpuPowerPreference::HighPerformance);

        let adapter_promise = gpu.request_adapter_with_options(&adapter_options);
        let adapter_result = wasm_bindgen_futures::JsFuture::from(adapter_promise)
            .await
            .map_err(|_| TensorError::device_error_simple("Failed to request WebGPU adapter"))?;

        let adapter = adapter_result
            .dyn_into::<web_sys::GpuAdapter>()
            .map_err(|_| TensorError::device_error_simple("Invalid adapter object"))?;

        // Get adapter info
        let features = adapter.features();
        let limits = adapter.limits();

        // Request device
        let device_descriptor = web_sys::GpuDeviceDescriptor::new();

        let device_promise = adapter.request_device_with_descriptor(&device_descriptor);
        let device_result = wasm_bindgen_futures::JsFuture::from(device_promise)
            .await
            .map_err(|_| TensorError::device_error_simple("Failed to request WebGPU device"))?;

        let device = device_result
            .dyn_into::<web_sys::GpuDevice>()
            .map_err(|_| TensorError::device_error_simple("Invalid device object"))?;

        let queue = device.queue();

        self.adapter = Some(adapter);
        self.device = Some(device);
        self.queue = Some(queue);
        self.supported_features = Some(features);
        self.limits = Some(limits);

        Ok(())
    }

    /// Check if WebGPU is available
    pub fn is_available() -> bool {
        if let Some(window) = web_sys::window() {
            if let Some(_gpu) = window.navigator().gpu() {
                return true;
            }
        }
        false
    }

    /// Create a compute shader
    fn create_shader(&self, source: &str) -> Result<web_sys::GpuShaderModule> {
        let device = self
            .device
            .as_ref()
            .ok_or_else(|| TensorError::device_error_simple("WebGPU device not initialized"))?;

        let shader_descriptor = web_sys::GpuShaderModuleDescriptor::new(source);
        let shader = device.create_shader_module(&shader_descriptor);

        Ok(shader)
    }

    /// Create a compute pipeline
    fn create_compute_pipeline(
        &self,
        shader: &web_sys::GpuShaderModule,
        entry_point: &str,
    ) -> Result<web_sys::GpuComputePipeline> {
        let device = self
            .device
            .as_ref()
            .ok_or_else(|| TensorError::device_error_simple("WebGPU device not initialized"))?;

        let compute_stage = web_sys::GpuProgrammableStage::new(shader, entry_point);
        let pipeline_descriptor = web_sys::GpuComputePipelineDescriptor::new(&compute_stage);

        let pipeline = device.create_compute_pipeline(&pipeline_descriptor);

        Ok(pipeline)
    }

    /// Create a buffer
    fn create_buffer(&self, size: u64, usage: u32) -> Result<web_sys::GpuBuffer> {
        let device = self
            .device
            .as_ref()
            .ok_or_else(|| TensorError::device_error_simple("WebGPU device not initialized"))?;

        let buffer_descriptor = web_sys::GpuBufferDescriptor::new(size, usage);
        let buffer = device.create_buffer(&buffer_descriptor);

        Ok(buffer)
    }

    /// Execute tensor addition on GPU
    pub async fn add_gpu(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        if a.len() != b.len() {
            return Err(TensorError::invalid_shape_simple(
                "Arrays must have same length",
            ));
        }

        let length = a.len();
        let byte_size = (length * 4) as u64; // f32 = 4 bytes

        // Create buffers
        let input_buffer_a = self.create_buffer(
            byte_size,
            web_sys::gpu_buffer_usage::STORAGE | web_sys::gpu_buffer_usage::COPY_DST,
        )?;
        let input_buffer_b = self.create_buffer(
            byte_size,
            web_sys::gpu_buffer_usage::STORAGE | web_sys::gpu_buffer_usage::COPY_DST,
        )?;
        let output_buffer = self.create_buffer(
            byte_size,
            web_sys::gpu_buffer_usage::STORAGE | web_sys::gpu_buffer_usage::COPY_SRC,
        )?;
        let staging_buffer = self.create_buffer(
            byte_size,
            web_sys::gpu_buffer_usage::MAP_READ | web_sys::gpu_buffer_usage::COPY_DST,
        )?;

        let queue = self
            .queue
            .as_ref()
            .ok_or_else(|| TensorError::device_error_simple("WebGPU queue not available"))?;

        // Copy data to GPU buffers
        let a_bytes = bytemuck::cast_slice(a);
        let b_bytes = bytemuck::cast_slice(b);

        queue.write_buffer_with_u8_array(&input_buffer_a, 0, a_bytes);
        queue.write_buffer_with_u8_array(&input_buffer_b, 0, b_bytes);

        // Get or create shader
        let shader_source = self.generate_add_shader(length);
        let shader = if let Some(cached_shader) = self.shader_cache.borrow().get(&shader_source) {
            cached_shader.clone()
        } else {
            let new_shader = self.create_shader(&shader_source)?;
            self.shader_cache
                .borrow_mut()
                .insert(shader_source.clone(), new_shader.clone());
            new_shader
        };

        // Get or create compute pipeline
        let pipeline_key = format!("add_{}", length);
        let pipeline = if let Some(cached_pipeline) =
            self.compute_pipeline_cache.borrow().get(&pipeline_key)
        {
            cached_pipeline.clone()
        } else {
            let new_pipeline = self.create_compute_pipeline(&shader, "main")?;
            self.compute_pipeline_cache
                .borrow_mut()
                .insert(pipeline_key, new_pipeline.clone());
            new_pipeline
        };

        // Create bind group
        let device = self
            .device
            .as_ref()
            .ok_or_else(|| TensorError::ComputeError {
                operation: "wasm_gpu_operation".to_string(),
                details: "WebGPU device not initialized".to_string(),
                retry_possible: false,
                context: None,
            })?;
        let bind_group_layout = pipeline.get_bind_group_layout(0);

        let bind_group_entries = js_sys::Array::new();

        // Binding 0: input buffer A
        let entry_a = web_sys::GpuBindGroupEntry::new(0, &input_buffer_a);
        bind_group_entries.push(&entry_a);

        // Binding 1: input buffer B
        let entry_b = web_sys::GpuBindGroupEntry::new(1, &input_buffer_b);
        bind_group_entries.push(&entry_b);

        // Binding 2: output buffer
        let entry_output = web_sys::GpuBindGroupEntry::new(2, &output_buffer);
        bind_group_entries.push(&entry_output);

        let bind_group_descriptor =
            web_sys::GpuBindGroupDescriptor::new(&bind_group_entries, &bind_group_layout);
        let bind_group = device.create_bind_group(&bind_group_descriptor);

        // Create command encoder
        let command_encoder = device.create_command_encoder();
        let compute_pass = command_encoder.begin_compute_pass();

        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, Some(&bind_group));

        let workgroup_size = 64;
        let num_workgroups = (length + workgroup_size - 1) / workgroup_size;
        compute_pass.dispatch_workgroups(num_workgroups as u32);
        compute_pass.end();

        // Copy result to staging buffer
        command_encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, byte_size);

        let command_buffer = command_encoder.finish();
        queue.submit(&js_sys::Array::of1(&command_buffer));

        // Map and read result
        let map_promise = staging_buffer.map_async(web_sys::gpu_map_mode::READ, 0, byte_size);
        wasm_bindgen_futures::JsFuture::from(map_promise)
            .await
            .map_err(|_| TensorError::device_error_simple("Failed to map staging buffer"))?;

        let mapped_range = staging_buffer.get_mapped_range_with_f64_and_f64(0.0, byte_size as f64);
        let result_bytes = js_sys::Uint8Array::new(&mapped_range);

        let mut result_data = vec![0u8; byte_size as usize];
        result_bytes.copy_to(&mut result_data);

        staging_buffer.unmap();

        // Convert bytes back to f32 array
        let result: Vec<f32> = bytemuck::cast_slice(&result_data).to_vec();

        Ok(result)
    }

    /// Generate WebGPU compute shader for addition
    fn generate_add_shader(&self, length: usize) -> String {
        format!(
            r#"
@group(0) @binding(0) var<storage, read> input_a: array<f32>;
@group(0) @binding(1) var<storage, read> input_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let index = global_id.x;
    if (index >= {}u) {{
        return;
    }}

    output[index] = input_a[index] + input_b[index];
}}
"#,
            length
        )
    }

    /// Execute tensor multiplication on GPU
    pub async fn mul_gpu(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        if a.len() != b.len() {
            return Err(TensorError::invalid_shape_simple(
                "Arrays must have same length",
            ));
        }

        // Similar implementation to add_gpu but with multiplication shader
        let length = a.len();
        let shader_source = format!(
            r#"
@group(0) @binding(0) var<storage, read> input_a: array<f32>;
@group(0) @binding(1) var<storage, read> input_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let index = global_id.x;
    if (index >= {}u) {{
        return;
    }}

    output[index] = input_a[index] * input_b[index];
}}
"#,
            length
        );

        // Reusing the same execution pattern as add_gpu
        self.execute_binary_op(a, b, &shader_source, "mul").await
    }

    /// Generic binary operation execution
    async fn execute_binary_op(
        &self,
        a: &[f32],
        b: &[f32],
        shader_source: &str,
        op_name: &str,
    ) -> Result<Vec<f32>> {
        let length = a.len();
        let byte_size = (length * 4) as u64;

        // Create buffers (same as add_gpu)
        let input_buffer_a = self.create_buffer(
            byte_size,
            web_sys::gpu_buffer_usage::STORAGE | web_sys::gpu_buffer_usage::COPY_DST,
        )?;
        let input_buffer_b = self.create_buffer(
            byte_size,
            web_sys::gpu_buffer_usage::STORAGE | web_sys::gpu_buffer_usage::COPY_DST,
        )?;
        let output_buffer = self.create_buffer(
            byte_size,
            web_sys::gpu_buffer_usage::STORAGE | web_sys::gpu_buffer_usage::COPY_SRC,
        )?;
        let staging_buffer = self.create_buffer(
            byte_size,
            web_sys::gpu_buffer_usage::MAP_READ | web_sys::gpu_buffer_usage::COPY_DST,
        )?;

        let queue = self
            .queue
            .as_ref()
            .ok_or_else(|| TensorError::device_error_simple("WebGPU queue not available"))?;

        // Upload data
        let a_bytes = bytemuck::cast_slice(a);
        let b_bytes = bytemuck::cast_slice(b);

        queue.write_buffer_with_u8_array(&input_buffer_a, 0, a_bytes);
        queue.write_buffer_with_u8_array(&input_buffer_b, 0, b_bytes);

        // Create or get cached shader and pipeline
        let shader = self.create_shader(shader_source)?;
        let pipeline = self.create_compute_pipeline(&shader, "main")?;

        // Execute compute pass
        let device = self
            .device
            .as_ref()
            .ok_or_else(|| TensorError::device_error_simple("WebGPU device not initialized"))?;
        let bind_group_layout = pipeline.get_bind_group_layout(0);

        let bind_group_entries = js_sys::Array::new();
        bind_group_entries.push(&web_sys::GpuBindGroupEntry::new(0, &input_buffer_a));
        bind_group_entries.push(&web_sys::GpuBindGroupEntry::new(1, &input_buffer_b));
        bind_group_entries.push(&web_sys::GpuBindGroupEntry::new(2, &output_buffer));

        let bind_group_descriptor =
            web_sys::GpuBindGroupDescriptor::new(&bind_group_entries, &bind_group_layout);
        let bind_group = device.create_bind_group(&bind_group_descriptor);

        let command_encoder = device.create_command_encoder();
        let compute_pass = command_encoder.begin_compute_pass();

        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, Some(&bind_group));

        let workgroup_size = 64;
        let num_workgroups = (length + workgroup_size - 1) / workgroup_size;
        compute_pass.dispatch_workgroups(num_workgroups as u32);
        compute_pass.end();

        command_encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, byte_size);

        let command_buffer = command_encoder.finish();
        queue.submit(&js_sys::Array::of1(&command_buffer));

        // Read result
        let map_promise = staging_buffer.map_async(web_sys::gpu_map_mode::READ, 0, byte_size);
        wasm_bindgen_futures::JsFuture::from(map_promise)
            .await
            .map_err(|_| TensorError::device_error_simple("Failed to map staging buffer"))?;

        let mapped_range = staging_buffer.get_mapped_range_with_f64_and_f64(0.0, byte_size as f64);
        let result_bytes = js_sys::Uint8Array::new(&mapped_range);

        let mut result_data = vec![0u8; byte_size as usize];
        result_bytes.copy_to(&mut result_data);

        staging_buffer.unmap();

        let result: Vec<f32> = bytemuck::cast_slice(&result_data).to_vec();
        Ok(result)
    }

    /// Get WebGPU device limits
    pub fn get_limits(&self) -> Option<WebGpuLimits> {
        self.limits.as_ref().map(|limits| WebGpuLimits {
            max_texture_dimension_1d: limits.max_texture_dimension_1d(),
            max_texture_dimension_2d: limits.max_texture_dimension_2d(),
            max_texture_dimension_3d: limits.max_texture_dimension_3d(),
            max_bind_groups: limits.max_bind_groups(),
            max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size() as usize,
            max_compute_workgroup_size_x: limits.max_compute_workgroup_size_x(),
            max_compute_workgroup_size_y: limits.max_compute_workgroup_size_y(),
            max_compute_workgroup_size_z: limits.max_compute_workgroup_size_z(),
            max_compute_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension(),
        })
    }

    /// Check if a feature is supported
    pub fn has_feature(&self, feature: &str) -> bool {
        if let Some(features) = &self.supported_features {
            features.has(feature)
        } else {
            false
        }
    }
}

// ================================
// WasmContextWithGpu implementation
// ================================

#[cfg(target_arch = "wasm32")]
impl WasmContextWithGpu {
    /// Create new context with WebGPU support
    pub fn new() -> Self {
        Self {
            cpu_ops: WasmTensorOps::new(),
            gpu_backend: None,
            prefer_gpu: true,
            gpu_threshold: 1024, // Use GPU for arrays larger than 1024 elements
        }
    }

    /// Initialize WebGPU backend
    pub async fn init_gpu(&mut self) -> Result<()> {
        if WebGpuBackend::is_available() {
            let mut backend = WebGpuBackend::new();
            backend.initialize().await?;
            self.gpu_backend = Some(backend);
            Ok(())
        } else {
            Err(TensorError::device_error_simple("WebGPU not available"))
        }
    }

    /// Perform addition with automatic CPU/GPU selection
    pub async fn add_auto(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        if self.should_use_gpu(a.len()) {
            if let Some(gpu) = &self.gpu_backend {
                return gpu.add_gpu(a, b).await;
            }
        }

        // Fallback to CPU SIMD
        let mut result = vec![0.0; a.len()];
        self.cpu_ops.add_simd(a, b, &mut result)?;
        Ok(result)
    }

    /// Perform multiplication with automatic CPU/GPU selection
    pub async fn mul_auto(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        if self.should_use_gpu(a.len()) {
            if let Some(gpu) = &self.gpu_backend {
                return gpu.mul_gpu(a, b).await;
            }
        }

        // Fallback to CPU SIMD
        let mut result = vec![0.0; a.len()];
        self.cpu_ops.mul_simd(a, b, &mut result)?;
        Ok(result)
    }

    /// Check if GPU should be used for given array size
    fn should_use_gpu(&self, size: usize) -> bool {
        self.prefer_gpu && self.gpu_backend.is_some() && size >= self.gpu_threshold
    }

    /// Get GPU backend info
    pub fn gpu_info(&self) -> Option<String> {
        self.gpu_backend.as_ref().map(|gpu| {
            let limits = gpu.get_limits().unwrap_or_else(|| WebGpuLimits {
                max_texture_dimension_1d: 0,
                max_texture_dimension_2d: 0,
                max_texture_dimension_3d: 0,
                max_bind_groups: 0,
                max_storage_buffer_binding_size: 0,
                max_compute_workgroup_size_x: 0,
                max_compute_workgroup_size_y: 0,
                max_compute_workgroup_size_z: 0,
                max_compute_workgroups_per_dimension: 0,
            });

            format!(
                "WebGPU Backend Active - Max Buffer Size: {} MB, Max Workgroup Size: {}x{}x{}, Max Workgroups: {}",
                limits.max_storage_buffer_binding_size / (1024 * 1024),
                limits.max_compute_workgroup_size_x,
                limits.max_compute_workgroup_size_y,
                limits.max_compute_workgroup_size_z,
                limits.max_compute_workgroups_per_dimension
            )
        })
    }

    /// Set GPU usage threshold
    pub fn set_gpu_threshold(&mut self, threshold: usize) {
        self.gpu_threshold = threshold;
    }

    /// Enable or disable GPU preference
    pub fn set_prefer_gpu(&mut self, prefer: bool) {
        self.prefer_gpu = prefer;
    }
}
