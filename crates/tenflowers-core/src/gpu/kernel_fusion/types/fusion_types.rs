//! FusedOperation — the central type for kernel fusion.

use std::collections::HashMap;

use super::config::{HardwareConfig, MemoryAccessPattern, SimdConfig, SimdInstructionSet};
use super::kernel_types::{FusableOp, ParallelizationStrategy};
use super::metrics::PerformanceProfile;
use super::patterns::GpuVendorHints;

/// Ultra-performance fused operation sequence with advanced optimization
#[derive(Debug, Clone)]
pub struct FusedOperation {
    /// Sequence of operations to fuse
    pub operations: Vec<FusableOp>,
    /// Operation parameters (e.g., epsilon for BatchNorm)
    pub parameters: HashMap<String, f32>,
    /// Input tensor count
    pub input_count: usize,
    /// Output tensor count
    pub output_count: usize,
    /// Kernel identifier for shader selection
    pub kernel_id: String,
    /// GPU vendor-specific optimization hints
    pub vendor_hints: GpuVendorHints,
    /// Memory access patterns for bandwidth optimization
    pub memory_patterns: MemoryAccessPattern,
    /// SIMD vectorization configuration
    pub simd_config: SimdConfig,
    /// Tensor Core optimization settings
    pub hardware_config: Option<HardwareConfig>,
    /// Performance characteristics for ML-based optimization
    pub perf_profile: PerformanceProfile,
    /// Fusion priority score (higher = more beneficial)
    pub fusion_priority: f64,
    /// Expected memory bandwidth reduction
    pub bandwidth_reduction: f64,
    /// Parallel execution hints
    pub parallelization_strategy: ParallelizationStrategy,
}

impl FusedOperation {
    /// Create a new fused operation
    pub fn new(operations: Vec<FusableOp>) -> Self {
        let input_count = Self::calculate_input_count(&operations);
        let output_count = 1;
        let kernel_id = Self::generate_kernel_id(&operations);
        Self {
            operations,
            parameters: HashMap::new(),
            input_count,
            output_count,
            kernel_id,
            vendor_hints: GpuVendorHints::Generic,
            memory_patterns: MemoryAccessPattern::Sequential,
            simd_config: SimdConfig {
                vector_width: 4,
                enable_vectorization: true,
                instruction_set: SimdInstructionSet::Avx2,
                alignment: 16,
            },
            hardware_config: None,
            perf_profile: PerformanceProfile {
                estimated_flops: 1000000,
                memory_bandwidth: 1000000000,
                arithmetic_intensity: 1.0,
                estimated_latency: 100.0,
                cache_efficiency: 0.8,
                parallel_efficiency: 0.9,
                historical_performance: Vec::new(),
            },
            fusion_priority: 1.0,
            bandwidth_reduction: 0.1,
            parallelization_strategy: ParallelizationStrategy::DataParallel { num_devices: 1 },
        }
    }

    /// Create fused MatMul + Bias + Activation
    pub fn fused_dense_layer(activation: Option<FusableOp>) -> Self {
        let mut ops = vec![FusableOp::MatMul, FusableOp::Add];
        if let Some(act) = activation {
            ops.push(act);
        }
        Self::new(ops)
    }

    /// Create fused Element-wise + Activation
    pub fn fused_elementwise_activation(elementwise_op: FusableOp, activation: FusableOp) -> Self {
        Self::new(vec![elementwise_op, activation])
    }

    /// Create fused Convolution + BatchNorm + Activation
    pub fn fused_conv_bn_activation(activation: FusableOp) -> Self {
        let mut ops = vec![FusableOp::Conv2D, FusableOp::BatchNorm];
        ops.push(activation);
        Self::new(ops)
    }

    /// Create ultra-optimized Flash Attention fusion for transformers
    pub fn fused_flash_attention() -> Self {
        Self::new(vec![
            FusableOp::MatMul,
            FusableOp::Mul,
            FusableOp::Softmax,
            FusableOp::MatMul,
        ])
    }

    /// Create fused RMSNorm + Linear + Activation for modern transformers
    pub fn fused_rmsnorm_linear_activation(activation: FusableOp) -> Self {
        Self::new(vec![
            FusableOp::RMSNorm,
            FusableOp::MatMul,
            FusableOp::Add,
            activation,
        ])
    }

    /// Create fused SwiGLU operation (used in LLaMA, PaLM)
    pub fn fused_swiglu() -> Self {
        Self::new(vec![
            FusableOp::MatMul,
            FusableOp::MatMul,
            FusableOp::Swish,
            FusableOp::Mul,
        ])
    }

    /// Create fused GeGLU operation (used in T5, PaLM)
    pub fn fused_geglu() -> Self {
        Self::new(vec![
            FusableOp::MatMul,
            FusableOp::MatMul,
            FusableOp::GELU,
            FusableOp::Mul,
        ])
    }

    /// Create fused quantized linear layer for inference optimization
    pub fn fused_quantized_linear(quantization_bits: u8) -> Self {
        let dequant_op = match quantization_bits {
            4 => FusableOp::Dequantize4,
            8 => FusableOp::Dequantize8,
            _ => FusableOp::Dequantize8,
        };
        Self::new(vec![dequant_op, FusableOp::MatMul, FusableOp::Add])
    }

    /// Create fused FP8 operations for latest Hopper/Ada architectures
    pub fn fused_fp8_linear() -> Self {
        Self::new(vec![FusableOp::FP8MatMul, FusableOp::FP8Add])
    }

    /// Create fused MoE (Mixture of Experts) gate computation
    pub fn fused_moe_gating() -> Self {
        Self::new(vec![FusableOp::MatMul, FusableOp::Softmax])
    }

    /// Create fused depthwise separable convolution
    pub fn fused_depthwise_separable_conv(activation: FusableOp) -> Self {
        Self::new(vec![
            FusableOp::DepthwiseConv2D,
            FusableOp::BatchNorm,
            activation,
            FusableOp::Conv2D,
            FusableOp::BatchNorm,
            activation,
        ])
    }

    /// Create fused Multi-Head Attention pattern (Q*K^T + softmax + *V)
    pub fn fused_multihead_attention() -> Self {
        Self::new(vec![FusableOp::MatMul, FusableOp::Add, FusableOp::MatMul])
            .with_parameter("scale".to_string(), 1.0)
    }

    /// Create fused Residual Connection (input + layer(input))
    pub fn fused_residual_connection(inner_ops: Vec<FusableOp>) -> Self {
        let mut ops = inner_ops;
        ops.push(FusableOp::Add);
        Self::new(ops)
    }

    /// Create fused Layer Normalization + Linear + Activation
    pub fn fused_layernorm_linear_activation(activation: FusableOp) -> Self {
        Self::new(vec![
            FusableOp::LayerNorm,
            FusableOp::MatMul,
            FusableOp::Add,
            activation,
        ])
    }

    /// Create fused GELU approximation (x * 0.5 * (1 + tanh(...)))
    pub fn fused_gelu_approximation() -> Self {
        Self::new(vec![
            FusableOp::Mul,
            FusableOp::Add,
            FusableOp::Tanh,
            FusableOp::Mul,
        ])
        .with_parameter("gelu_coeff".to_string(), 0.044715)
    }

    /// Create fused Dropout + Scale (for training efficiency)
    pub fn fused_dropout_scale(dropout_rate: f32) -> Self {
        Self::new(vec![FusableOp::Mul])
            .with_parameter("dropout_rate".to_string(), dropout_rate)
            .with_parameter("scale_factor".to_string(), 1.0 / (1.0 - dropout_rate))
    }

    /// Create fused Swish/SiLU activation (x * sigmoid(x))
    pub fn fused_swish_activation() -> Self {
        Self::new(vec![FusableOp::Sigmoid, FusableOp::Mul])
    }

    /// Create fused Element-wise operations chain (optimized for common patterns)
    pub fn fused_elementwise_chain(ops: Vec<FusableOp>) -> Self {
        for op in &ops {
            match op {
                FusableOp::Add
                | FusableOp::Mul
                | FusableOp::Sub
                | FusableOp::Div
                | FusableOp::ReLU
                | FusableOp::Sigmoid
                | FusableOp::Tanh
                | FusableOp::GELU
                | FusableOp::Swish => {}
                _ => panic!("Only element-wise operations allowed in element-wise chain"),
            }
        }
        Self::new(ops)
    }

    /// Advanced fusion for transformer feed-forward network
    pub fn fused_transformer_ffn() -> Self {
        Self::new(vec![
            FusableOp::LayerNorm,
            FusableOp::MatMul,
            FusableOp::Add,
            FusableOp::GELU,
            FusableOp::MatMul,
            FusableOp::Add,
        ])
    }

    /// Check if operations can be safely fused together
    pub fn can_fuse_operations(ops: &[FusableOp]) -> bool {
        if ops.is_empty() || ops.len() > 8 {
            return false;
        }
        let has_batch_norm = ops.contains(&FusableOp::BatchNorm);
        let has_layer_norm = ops.contains(&FusableOp::LayerNorm);
        if ops.iter().filter(|&&op| op == FusableOp::MatMul).count() > 2 {
            return false;
        }
        if has_batch_norm && has_layer_norm {
            return false;
        }
        true
    }

    /// Estimate performance benefit of fusion
    pub fn estimate_fusion_benefit(&self) -> f32 {
        let base_benefit = match self.operations.len() {
            0..=1 => 0.0,
            2 => 1.5,
            3 => 2.2,
            4 => 2.8,
            5..=6 => 3.5,
            _ => 4.0,
        };
        let memory_bandwidth_bonus = if self.operations.iter().any(|op| {
            matches!(
                op,
                FusableOp::MatMul | FusableOp::BatchNorm | FusableOp::LayerNorm
            )
        }) {
            1.3
        } else {
            1.0
        };
        let complexity_penalty = if self.operations.len() > 6 { 0.8 } else { 1.0 };
        base_benefit * memory_bandwidth_bonus * complexity_penalty
    }

    /// Add parameter to the fused operation
    pub fn with_parameter(mut self, key: String, value: f32) -> Self {
        self.parameters.insert(key, value);
        self
    }

    /// Calculate input count based on operations
    fn calculate_input_count(operations: &[FusableOp]) -> usize {
        if operations.contains(&FusableOp::MatMul) {
            3
        } else if operations.len() >= 2
            && matches!(
                operations[0],
                FusableOp::Add | FusableOp::Mul | FusableOp::Sub | FusableOp::Div
            )
        {
            2
        } else {
            1
        }
    }

    /// Generate unique kernel identifier
    pub fn generate_kernel_id(operations: &[FusableOp]) -> String {
        let op_names: Vec<String> = operations
            .iter()
            .map(|op| format!("{:?}", op).to_lowercase())
            .collect();
        format!("fused_{}", op_names.join("_"))
    }
}
