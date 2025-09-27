//! WASM-optimized tensor operations with minimal memory footprint

#[cfg(feature = "wasm")]
use crate::{Result, TensorError};
#[cfg(feature = "wasm")]
use std::collections::HashMap;
#[cfg(feature = "wasm")]
use std::hash::Hash;

#[cfg(feature = "wasm")]
use js_sys::*;
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;
#[cfg(feature = "wasm")]
use web_sys::*;

#[cfg(feature = "wasm")]
use super::compression::{
    WasmCompressedData, WasmQuantizedData, WasmRunLengthData, WasmSparseData,
};

/// WASM-optimized tensor operations with minimal memory footprint
#[cfg(feature = "wasm")]
pub struct WasmOptimizedTensor<T> {
    /// Compressed data storage
    data: WasmCompressedData<T>,
    /// Shape information
    shape: Vec<usize>,
    /// Memory layout optimization flags
    layout_flags: WasmLayoutFlags,
}

/// Memory layout optimization flags
#[cfg(feature = "wasm")]
#[derive(Debug, Clone, Copy)]
pub struct WasmLayoutFlags {
    /// Use memory-mapped storage
    pub memory_mapped: bool,
    /// Enable SIMD optimizations
    pub simd_enabled: bool,
    /// Use SharedArrayBuffer if available
    pub shared_buffer: bool,
    /// Enable streaming computation
    pub streaming: bool,
}

#[cfg(feature = "wasm")]
impl<T> WasmOptimizedTensor<T>
where
    T: Clone + Default + PartialEq,
{
    /// Create a new optimized tensor for WASM deployment
    pub fn new(data: Vec<T>, shape: Vec<usize>) -> Result<Self> {
        let layout_flags = WasmLayoutFlags {
            memory_mapped: false,
            simd_enabled: Self::detect_simd_support(),
            shared_buffer: Self::detect_shared_buffer_support(),
            streaming: false,
        };

        // Choose optimal storage format based on data characteristics
        let compressed_data = Self::choose_optimal_storage(&data)?;

        Ok(WasmOptimizedTensor {
            data: compressed_data,
            shape,
            layout_flags,
        })
    }

    /// Detect SIMD support in current WASM environment
    fn detect_simd_support() -> bool {
        // Check for WASM SIMD support
        #[cfg(target_arch = "wasm32")]
        {
            // In practice, this would check for actual SIMD instruction support
            true
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            false
        }
    }

    /// Detect SharedArrayBuffer support
    fn detect_shared_buffer_support() -> bool {
        #[cfg(feature = "wasm")]
        {
            // Check if SharedArrayBuffer is available
            js_sys::eval("typeof SharedArrayBuffer !== 'undefined'")
                .map(|val| val.as_bool().unwrap_or(false))
                .unwrap_or(false)
        }
        #[cfg(not(feature = "wasm"))]
        {
            false
        }
    }

    /// Choose optimal storage format based on data characteristics
    fn choose_optimal_storage(data: &[T]) -> Result<WasmCompressedData<T>> {
        let data_size = data.len();
        let unique_values = Self::count_unique_values(data);
        let sparsity = Self::calculate_sparsity(data);

        // Decision logic for storage format
        if sparsity > 0.9 && data_size > 1000 {
            // Use sparse storage for highly sparse large tensors
            Ok(WasmCompressedData::Sparse(Self::create_sparse_data(data)?))
        } else if unique_values < data_size / 10 {
            // Use run-length encoding for repetitive data
            Ok(WasmCompressedData::RunLength(Self::create_run_length_data(
                data,
            )))
        } else if data_size > 10000 {
            // Use quantization for large dense tensors
            Ok(WasmCompressedData::Quantized(Self::create_quantized_data(
                data,
            )?))
        } else {
            // Use dense storage for small tensors
            Ok(WasmCompressedData::Dense(data.to_vec()))
        }
    }

    fn count_unique_values(data: &[T]) -> usize {
        // For floating-point types, we can't use HashSet due to NaN issues
        // Use a simple O(n²) approach for uniqueness counting
        let mut unique_items = Vec::new();
        for item in data {
            if !unique_items.iter().any(|existing| *existing == item) {
                unique_items.push(item);
            }
        }
        unique_items.len()
    }

    fn calculate_sparsity(data: &[T]) -> f64 {
        let zero_count = data.iter().filter(|&&ref x| *x == T::default()).count();
        zero_count as f64 / data.len() as f64
    }

    fn create_sparse_data(data: &[T]) -> Result<WasmSparseData<T>> {
        // Create CSR sparse representation
        let mut values = Vec::new();
        let mut col_indices = Vec::new();
        let mut row_ptr = vec![0];

        let width = (data.len() as f64).sqrt() as usize; // Assume square matrix for simplicity
        let height = (data.len() + width - 1) / width;

        for i in 0..height {
            for j in 0..width {
                let idx = i * width + j;
                if idx < data.len() && data[idx] != T::default() {
                    values.push(data[idx].clone());
                    col_indices.push(j as u32);
                }
            }

            row_ptr.push(values.len() as u32);
        }

        let nnz = values.len();
        Ok(WasmSparseData {
            values,
            row_ptr,
            col_indices,
            nnz,
        })
    }

    fn create_run_length_data(data: &[T]) -> WasmRunLengthData<T> {
        let mut values = Vec::new();
        let mut lengths = Vec::new();

        if data.is_empty() {
            return WasmRunLengthData { values, lengths };
        }

        let mut current_value = &data[0];
        let mut current_length = 1u32;

        for item in data.iter().skip(1) {
            if item == current_value {
                current_length += 1;
            } else {
                values.push(current_value.clone());
                lengths.push(current_length);
                current_value = item;
                current_length = 1;
            }
        }

        // Add final run
        values.push(current_value.clone());
        lengths.push(current_length);

        WasmRunLengthData { values, lengths }
    }

    fn create_quantized_data(data: &[T]) -> Result<WasmQuantizedData> {
        // Simplified quantization for demonstration
        // In practice, this would implement proper quantization schemes

        let quantized_values = vec![0u8; data.len()]; // Placeholder

        Ok(WasmQuantizedData {
            quantized_values,
            scale: 1.0,
            zero_point: 0,
            bit_width: 8,
        })
    }

    /// Get tensor shape
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        match &self.data {
            WasmCompressedData::Dense(data) => data.len() * std::mem::size_of::<T>(),
            WasmCompressedData::Sparse(sparse) => {
                sparse.values.len() * std::mem::size_of::<T>()
                    + sparse.col_indices.len() * 4
                    + sparse.row_ptr.len() * 4
            }
            WasmCompressedData::Quantized(quant) => quant.quantized_values.len() + 16,
            WasmCompressedData::RunLength(rle) => {
                rle.values.len() * std::mem::size_of::<T>() + rle.lengths.len() * 4
            }
        }
    }
}

/// WASM-specific operations for edge deployment
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmTensorOperations {
    memory_manager: super::memory::WasmMemoryManager,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmTensorOperations {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            memory_manager: super::memory::WasmMemoryManager::new(16 * 1024 * 1024), // 16MB limit
        }
    }

    /// Perform optimized matrix multiplication for WASM
    #[wasm_bindgen]
    pub fn matmul_optimized(
        &mut self,
        a: &[f32],
        b: &[f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Vec<f32> {
        // Optimized matrix multiplication for WASM
        let mut result = vec![0.0f32; m * n];

        // Use blocked algorithm for better cache performance
        const BLOCK_SIZE: usize = 32;

        for ii in (0..m).step_by(BLOCK_SIZE) {
            for jj in (0..n).step_by(BLOCK_SIZE) {
                for kk in (0..k).step_by(BLOCK_SIZE) {
                    for i in ii..std::cmp::min(ii + BLOCK_SIZE, m) {
                        for j in jj..std::cmp::min(jj + BLOCK_SIZE, n) {
                            for k_idx in kk..std::cmp::min(kk + BLOCK_SIZE, k) {
                                result[i * n + j] += a[i * k + k_idx] * b[k_idx * n + j];
                            }
                        }
                    }
                }
            }
        }

        result
    }

    /// Get memory usage statistics
    #[wasm_bindgen]
    pub fn get_memory_usage(&self) -> f64 {
        self.memory_manager.total_allocated as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "wasm")]
    #[ignore = "WASM tests require WASM target - cannot run on native"]
    fn test_wasm_tensor_optimization() {
        let data = vec![1.0f32, 0.0, 0.0, 2.0, 0.0];
        let shape = vec![5];

        let result = WasmOptimizedTensor::new(data, shape);
        assert!(result.is_ok());

        let tensor = result.unwrap();
        assert_eq!(tensor.shape(), &[5]);
    }
}
