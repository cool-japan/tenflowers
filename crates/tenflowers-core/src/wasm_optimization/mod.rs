//! WebAssembly optimization module for edge deployment
//!
//! This module provides optimizations specifically for WebAssembly builds,
//! focusing on minimal bundle sizes, efficient memory usage, and fast
//! initialization for edge computing and browser environments.
//!
//! ## Module Structure
//!
//! - [`tensor`]: WASM-optimized tensor operations with minimal memory footprint
//! - [`compression`]: Data compression formats for WASM-optimized tensors
//! - [`bundle`]: WASM bundle size optimization and configuration
//! - [`memory`]: WASM runtime memory management for edge deployment
//! - [`inference`]: Edge-optimized neural network inference for WASM deployment
//! - [`performance`]: Performance metrics and optimization reports for WASM deployment
//! - [`device`]: WASM device capabilities and platform detection

pub mod bundle;
pub mod compression;
pub mod device;
pub mod inference;
pub mod memory;
pub mod performance;
pub mod tensor;

// Re-export all public types for backward compatibility
#[cfg(feature = "wasm")]
pub use tensor::{WasmLayoutFlags, WasmOptimizedTensor, WasmTensorOperations};

#[cfg(feature = "wasm")]
pub use compression::{
    CompressionConfig, WasmCompressedData, WasmQuantizedData, WasmRunLengthData, WasmSparseData,
};

#[cfg(feature = "wasm")]
pub use bundle::{
    CodeSplittingConfig, TreeShakingConfig, WasmBundleOptimizer, WasmOptimizationConfig,
};

#[cfg(feature = "wasm")]
pub use memory::{WasmMemoryChunk, WasmMemoryManager, WasmMemoryStats};

#[cfg(feature = "wasm")]
pub use inference::{
    WasmActivationType, WasmBatchNormParams, WasmEdgeInference, WasmElementwiseOp,
    WasmFusedOperation, WasmInferenceCache, WasmModelMetadata, WasmModelStats, WasmOptimizedModel,
    WasmPrunedConnections,
};

#[cfg(feature = "wasm")]
pub use performance::{
    WasmBenchmarkConfig, WasmBenchmarkResult, WasmOptimizationReport, WasmPerformanceBenchmark,
    WasmPerformanceComparison, WasmPerformanceMetrics,
};

#[cfg(feature = "wasm")]
pub use device::{
    WasmDeviceCapabilities, WasmDeviceCategory, WasmDeviceInfo, WasmDeviceProfile,
    WasmDeviceProfiler, WasmFeatures, WasmOptimizationStrategy, WasmPerformanceTier,
    WasmProfileBenchmark, WasmRuntimeInfo, WasmVersion,
};

/// Stub implementation for non-WASM platforms
#[cfg(not(feature = "wasm"))]
pub mod wasm_stub {
    use crate::{Result, TensorError};

    pub fn wasm_not_available() -> Result<()> {
        Err(TensorError::device_error_simple(
            "WASM optimizations are only available with the 'wasm' feature enabled".to_string(),
        ))
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

    #[test]
    #[cfg(feature = "wasm")]
    fn test_bundle_optimizer() {
        let optimizer = WasmBundleOptimizer::new();
        assert!(optimizer.optimizations.dead_code_elimination);
        assert!(optimizer.optimizations.lto);
    }

    #[test]
    #[cfg(not(feature = "wasm"))]
    fn test_wasm_not_available() {
        let result = wasm_stub::wasm_not_available();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("WASM optimizations are only available"));
    }
}
