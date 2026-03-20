//! Core WebAssembly tensor operations
//!
//! This module provides the fundamental WebAssembly-specific tensor operations
//! with SIMD optimizations when available.

#[cfg(target_arch = "wasm32")]
use js_sys::{ArrayBuffer, Float32Array, Uint8Array, WebAssembly};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use web_sys::{console, window, Performance};

// WASM SIMD support
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use std::arch::wasm32::*;

use crate::{DType, Device, Result, TensorError};

/// WebAssembly-specific tensor operations
#[cfg(target_arch = "wasm32")]
pub struct WasmTensorOps {
    /// WebAssembly memory instance
    memory: Option<WebAssembly::Memory>,
    /// Performance interface for timing
    performance: Option<Performance>,
}

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

#[cfg(target_arch = "wasm32")]
impl Default for WasmTensorOps {
    fn default() -> Self {
        Self::new()
    }
}