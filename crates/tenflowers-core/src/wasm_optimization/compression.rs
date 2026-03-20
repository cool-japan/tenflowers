//! Data compression formats for WASM-optimized tensors

/// Compressed data storage for WASM deployment
#[cfg(feature = "wasm")]
#[derive(Debug)]
pub enum WasmCompressedData<T> {
    /// Dense storage for small tensors
    Dense(Vec<T>),
    /// Sparse storage for large sparse tensors
    Sparse(WasmSparseData<T>),
    /// Quantized storage for reduced precision
    Quantized(WasmQuantizedData),
    /// Run-length encoded for repetitive data
    RunLength(WasmRunLengthData<T>),
}

/// Sparse data representation optimized for WASM
#[cfg(feature = "wasm")]
#[derive(Debug)]
pub struct WasmSparseData<T> {
    /// Non-zero values
    pub values: Vec<T>,
    /// Row indices (CSR format)
    pub row_ptr: Vec<u32>,
    /// Column indices
    pub col_indices: Vec<u32>,
    /// Number of non-zero elements
    pub nnz: usize,
}

/// Quantized data for reduced memory usage
#[cfg(feature = "wasm")]
#[derive(Debug)]
pub struct WasmQuantizedData {
    /// Quantized values (8-bit or 16-bit)
    pub quantized_values: Vec<u8>,
    /// Scale factor for dequantization
    pub scale: f32,
    /// Zero point for quantization
    pub zero_point: i32,
    /// Quantization bit width
    pub bit_width: u8,
}

/// Run-length encoded data for repetitive patterns
#[cfg(feature = "wasm")]
#[derive(Debug)]
pub struct WasmRunLengthData<T> {
    /// Unique values
    pub values: Vec<T>,
    /// Run lengths
    pub lengths: Vec<u32>,
}

/// Compression configuration
#[cfg(feature = "wasm")]
#[derive(Debug, Clone)]
pub struct CompressionConfig {
    /// Enable Brotli compression
    pub brotli: bool,
    /// Enable Gzip compression
    pub gzip: bool,
    /// Compression level (1-9)
    pub compression_level: u8,
    /// Enable model weight compression
    pub compress_weights: bool,
}

#[cfg(feature = "wasm")]
impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            brotli: true,
            gzip: true,
            compression_level: 9,
            compress_weights: true,
        }
    }
}

#[cfg(feature = "wasm")]
impl CompressionConfig {
    /// Create new compression configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Create minimal compression configuration for size optimization
    pub fn minimal() -> Self {
        Self {
            brotli: true,
            gzip: false,
            compression_level: 9,
            compress_weights: true,
        }
    }

    /// Create fast compression configuration for speed optimization
    pub fn fast() -> Self {
        Self {
            brotli: false,
            gzip: true,
            compression_level: 1,
            compress_weights: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "wasm")]
    fn test_compression_config() {
        let config = CompressionConfig::new();
        assert!(config.brotli);
        assert_eq!(config.compression_level, 9);

        let minimal = CompressionConfig::minimal();
        assert!(minimal.brotli);
        assert!(!minimal.gzip);

        let fast = CompressionConfig::fast();
        assert!(!fast.brotli);
        assert!(fast.gzip);
        assert_eq!(fast.compression_level, 1);
    }
}
