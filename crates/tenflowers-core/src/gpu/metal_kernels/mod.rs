//! Metal Kernels for High-Performance GPU Operations
//!
//! This module provides comprehensive Metal compute kernels and Metal Performance Shaders (MPS)
//! integration for tensor operations on Apple devices, organized into specialized modules
//! for optimal maintainability and performance.
//!
//! ## Module Organization
//!
//! - [`device`] - Metal device management and pipeline caching
//! - [`types`] - Core types, enums, and configuration structures
//! - [`operations`] - MPS-based tensor operations (matmul, conv2d, reductions)
//! - [`benchmarks`] - Performance benchmarking and analysis tools
//! - [`mps_integration`] - Complete neural network operations with MPS
//! - [`shaders`] - Metal compute shader definitions
//!
//! ## Usage Example
//!
//! ```rust,no_run
//! # use tenflowers_core::gpu::metal_kernels::{MetalDevice, MetalKernelConfig};
//! # use tenflowers_core::Tensor;
//! # fn example() -> tenflowers_core::Result<()> {
//! // Create Metal device
//! let mut device = MetalDevice::new()?;
//!
//! // Execute optimized matrix multiplication
//! let a = Tensor::<f32>::zeros(&[512, 512])?;
//! let b = Tensor::<f32>::zeros(&[512, 512])?;
//! let result = device.matmul_mps(&a, &b)?;
//!
//! // Benchmark performance
//! let mut benchmark = MetalBenchmark::new()?;
//! let results = benchmark.benchmark_matmul(&[(256, 256, 256), (512, 512, 512)])?;
//! println!("{}", benchmark.generate_report());
//! # Ok(())
//! # }
//! ```

use crate::Result;

// Module declarations
pub mod benchmarks;
pub mod device;
pub mod mps_integration;
pub mod operations;
pub mod types;

// Shader resources
pub mod shaders {
    //! Metal compute shaders for tensor operations

    /// Embedded Metal kernel source code
    pub const METAL_KERNELS_SOURCE: &str = include_str!("shaders/metal_kernels.metal");
}

// Re-export core types and functionality
pub use device::{DeviceCapabilities, MetalDevice};
pub use types::{
    ActivationType, BenchmarkResult, ConvConfig, DispatchConfig, ElementwiseOp, LayerConfig,
    LayerType, MemoryAccessPattern, MetalKernelConfig, ReductionOp,
};

#[cfg(all(target_os = "macos", feature = "metal"))]
pub use benchmarks::{ConvConfig as BenchmarkConvConfig, MetalBenchmark};

#[cfg(all(target_os = "macos", feature = "metal"))]
pub use mps_integration::{LayerConfig as MPSLayerConfig, LayerType as MPSLayerType, MPSNeuralOps};

// Stub implementation for non-macOS platforms
#[cfg(not(all(target_os = "macos", feature = "metal")))]
pub mod metal_stub {
    //! Stub implementation for non-macOS platforms
    use crate::{Result, TensorError};

    /// Error message for Metal operations on non-macOS platforms
    pub fn metal_not_available() -> Result<()> {
        Err(TensorError::device_error_simple(
            "Metal kernels are only available on macOS with the 'metal' feature enabled"
                .to_string(),
        ))
    }

    /// Stub MetalDevice for non-macOS platforms
    pub struct MetalDevice;

    impl MetalDevice {
        pub fn new() -> Result<Self> {
            metal_not_available()?;
            Ok(MetalDevice)
        }
    }

    /// Stub MetalBenchmark for non-macOS platforms
    pub struct MetalBenchmark;

    impl MetalBenchmark {
        pub fn new() -> Result<Self> {
            metal_not_available()?;
            Ok(MetalBenchmark)
        }
    }

    /// Stub MPSNeuralOps for non-macOS platforms
    pub struct MPSNeuralOps;

    impl MPSNeuralOps {
        pub fn new() -> Result<Self> {
            metal_not_available()?;
            Ok(MPSNeuralOps)
        }
    }
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
pub use metal_stub::*;

/// High-level interface for Metal kernel operations
#[cfg(all(target_os = "macos", feature = "metal"))]
#[derive(Debug)]
pub struct MetalKernels {
    device: MetalDevice,
    benchmark: Option<MetalBenchmark>,
    mps_ops: Option<MPSNeuralOps>,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl MetalKernels {
    /// Create a new Metal kernels interface
    pub fn new() -> Result<Self> {
        Ok(MetalKernels {
            device: MetalDevice::new()?,
            benchmark: None,
            mps_ops: None,
        })
    }

    /// Get mutable reference to the Metal device
    pub fn device_mut(&mut self) -> &mut MetalDevice {
        &mut self.device
    }

    /// Get reference to the Metal device
    pub fn device(&self) -> &MetalDevice {
        &self.device
    }

    /// Initialize benchmarking capabilities
    pub fn with_benchmarking(mut self) -> Result<Self> {
        self.benchmark = Some(MetalBenchmark::new()?);
        Ok(self)
    }

    /// Initialize MPS neural network operations
    pub fn with_mps_ops(mut self) -> Result<Self> {
        self.mps_ops = Some(MPSNeuralOps::new()?);
        Ok(self)
    }

    /// Get mutable reference to benchmarking suite
    pub fn benchmark_mut(&mut self) -> Option<&mut MetalBenchmark> {
        self.benchmark.as_mut()
    }

    /// Get reference to benchmarking suite
    pub fn benchmark(&self) -> Option<&MetalBenchmark> {
        self.benchmark.as_ref()
    }

    /// Get mutable reference to MPS neural operations
    pub fn mps_ops_mut(&mut self) -> Option<&mut MPSNeuralOps> {
        self.mps_ops.as_mut()
    }

    /// Get reference to MPS neural operations
    pub fn mps_ops(&self) -> Option<&MPSNeuralOps> {
        self.mps_ops.as_ref()
    }

    /// Get device capabilities information
    pub fn get_capabilities(&self) -> DeviceCapabilities {
        self.device.get_device_capabilities()
    }

    /// Run comprehensive performance benchmarks
    pub fn run_benchmarks(&mut self) -> Result<Vec<BenchmarkResult>> {
        match self.benchmark.as_mut() {
            Some(benchmark) => benchmark.run_comprehensive_benchmarks(),
            None => Err(crate::TensorError::invalid_operation_simple(
                "Benchmarking not initialized. Call with_benchmarking() first".to_string(),
            )),
        }
    }

    /// Generate performance report
    pub fn generate_performance_report(&self) -> Result<String> {
        match self.benchmark.as_ref() {
            Some(benchmark) => Ok(benchmark.generate_report()),
            None => Err(crate::TensorError::invalid_operation_simple(
                "Benchmarking not initialized. Call with_benchmarking() first".to_string(),
            )),
        }
    }
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
#[derive(Debug)]
pub struct MetalKernels;

#[cfg(not(all(target_os = "macos", feature = "metal")))]
impl MetalKernels {
    pub fn new() -> Result<Self> {
        metal_stub::metal_not_available()?;
        Ok(MetalKernels)
    }
}

/// Convenience function to create a new Metal kernels interface
pub fn create_metal_kernels() -> Result<MetalKernels> {
    MetalKernels::new()
}

/// Convenience function to create Metal kernels with all features enabled
pub fn create_metal_kernels_full() -> Result<MetalKernels> {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        MetalKernels::new()?.with_benchmarking()?.with_mps_ops()
    }
    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    {
        MetalKernels::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn test_metal_kernels_creation() {
        let result = MetalKernels::new();
        // Test should pass on macOS with Metal support
        assert!(result.is_ok() || result.unwrap_err().to_string().contains("No Metal device"));
    }

    #[test]
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn test_metal_kernels_with_features() {
        if let Ok(mut kernels) = MetalKernels::new() {
            let result = kernels.with_benchmarking().and_then(|k| k.with_mps_ops());

            assert!(result.is_ok() || result.unwrap_err().to_string().contains("No Metal device"));
        }
    }

    #[test]
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn test_convenience_functions() {
        let basic = create_metal_kernels();
        let full = create_metal_kernels_full();

        // Either should succeed or fail consistently based on Metal availability
        if basic.is_ok() {
            assert!(full.is_ok());
        } else {
            assert!(full.is_err());
        }
    }

    #[test]
    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    fn test_metal_not_available() {
        let result = MetalKernels::new();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Metal kernels are only available on macOS"));
    }

    #[test]
    fn test_shader_source_inclusion() {
        // Verify that the shader source is properly embedded
        assert!(!shaders::METAL_KERNELS_SOURCE.is_empty());
        assert!(shaders::METAL_KERNELS_SOURCE.contains("elementwise_add"));
        assert!(shaders::METAL_KERNELS_SOURCE.contains("matrix_multiply_naive"));
    }

    #[test]
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn test_device_capabilities() {
        if let Ok(kernels) = MetalKernels::new() {
            let capabilities = kernels.get_capabilities();
            assert!(capabilities.max_threads_per_threadgroup > 0);
            assert!(capabilities.compute_units > 0);
            assert!(capabilities.memory_bandwidth_gbps > 0.0);
        }
    }
}
