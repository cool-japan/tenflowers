//! Metal Kernels Types and Configuration
//!
//! This module defines the core types, enums, and configuration structures
//! used throughout the Metal kernels system.

#[cfg(all(target_os = "macos", feature = "metal"))]
use metal;
use std::collections::HashMap;

/// Metal kernel execution configuration
#[cfg(all(target_os = "macos", feature = "metal"))]
#[derive(Debug, Clone)]
pub struct MetalKernelConfig {
    /// Thread group size (workgroup size)
    pub threads_per_group: metal::MTLSize,
    /// Number of thread groups to dispatch
    pub thread_groups: metal::MTLSize,
    // Memory barriers and synchronization
    // TODO: MTLBarrierScope is not available in metal crate v0.32.0
    // pub memory_barriers: Vec<metal::MTLBarrierScope>,
}

/// Reduction operation types
#[cfg(all(target_os = "macos", feature = "metal"))]
#[derive(Debug, Clone, Copy)]
pub enum ReductionOp {
    Sum,
    Mean,
    Max,
    Min,
}

/// Activation function types for fused kernels
#[cfg(all(target_os = "macos", feature = "metal"))]
#[derive(Debug, Clone, Copy)]
pub enum ActivationType {
    ReLU,
    GELU,
    Swish,
    Tanh,
    Sigmoid,
}

/// Element-wise operation types
#[cfg(all(target_os = "macos", feature = "metal"))]
#[derive(Debug, Clone, Copy)]
pub enum ElementwiseOp {
    Add,
    Mul,
    Sub,
    Div,
}

/// Layer types for neural network operations
#[cfg(all(target_os = "macos", feature = "metal"))]
#[derive(Debug, Clone)]
pub enum LayerType {
    Dense,
    Convolution,
    BatchNorm,
    LayerNorm,
    Activation,
}

/// Layer configuration for MPS operations
#[cfg(all(target_os = "macos", feature = "metal"))]
#[derive(Debug, Clone)]
pub struct LayerConfig {
    pub layer_type: LayerType,
    pub parameters: HashMap<String, Vec<f32>>,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
}

/// Convolution configuration for benchmarking
#[cfg(all(target_os = "macos", feature = "metal"))]
#[derive(Debug, Clone)]
pub struct ConvConfig {
    pub input_channels: usize,
    pub output_channels: usize,
    pub kernel_size: (usize, usize),
    pub stride: (usize, usize),
    pub padding: (usize, usize),
    pub input_size: (usize, usize),
}

/// Benchmark result for performance testing
#[cfg(all(target_os = "macos", feature = "metal"))]
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub operation: String,
    pub config: String,
    pub execution_time_ms: f64,
    pub throughput_gops: f64,
    pub memory_bandwidth_gbps: f64,
    pub efficiency_percent: f64,
}

/// Memory access patterns for optimization
#[cfg(all(target_os = "macos", feature = "metal"))]
#[derive(Debug, Clone)]
pub enum MemoryAccessPattern {
    Sequential,
    Strided { stride: usize },
    Tiled { tile_size: (usize, usize) },
    Blocked { block_size: usize },
}

/// Dispatch configuration for optimal kernel execution
#[cfg(all(target_os = "macos", feature = "metal"))]
#[derive(Debug, Clone)]
pub struct DispatchConfig {
    pub thread_groups: metal::MTLSize,
    pub threads_per_group: metal::MTLSize,
    pub memory_access: MemoryAccessPattern,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reduction_op_creation() {
        let op = ReductionOp::Sum;
        assert!(matches!(op, ReductionOp::Sum));
    }

    #[test]
    fn test_activation_type_creation() {
        let activation = ActivationType::ReLU;
        assert!(matches!(activation, ActivationType::ReLU));
    }

    #[test]
    fn test_elementwise_op_creation() {
        let op = ElementwiseOp::Add;
        assert!(matches!(op, ElementwiseOp::Add));
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn test_layer_config_creation() {
        use std::collections::HashMap;

        let mut parameters = HashMap::new();
        parameters.insert("weights".to_string(), vec![1.0, 2.0, 3.0]);

        let config = LayerConfig {
            layer_type: LayerType::Dense,
            parameters,
            input_shape: vec![128],
            output_shape: vec![64],
        };

        assert!(matches!(config.layer_type, LayerType::Dense));
        assert_eq!(config.input_shape, vec![128]);
        assert_eq!(config.output_shape, vec![64]);
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn test_conv_config_creation() {
        let config = ConvConfig {
            input_channels: 3,
            output_channels: 64,
            kernel_size: (3, 3),
            stride: (1, 1),
            padding: (1, 1),
            input_size: (224, 224),
        };

        assert_eq!(config.input_channels, 3);
        assert_eq!(config.output_channels, 64);
        assert_eq!(config.kernel_size, (3, 3));
    }
}
