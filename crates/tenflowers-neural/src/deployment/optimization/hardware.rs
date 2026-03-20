//! Hardware architecture definitions and configurations
//!
//! This module provides hardware architecture types and platform-specific
//! optimization configurations for different deployment targets.

use super::core::OptimizationConfig;
#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};
use tenflowers_core::DType;

/// Supported hardware architectures for auto-tuning.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum HardwareArchitecture {
    /// Apple Silicon (M1, M2, M3 series)
    AppleSilicon,
    /// NVIDIA GPU architectures
    NvidiaGPU { compute_capability: (u8, u8) },
    /// AMD GPU architectures
    AmdGPU { architecture: String },
    /// Intel GPU architectures
    IntelGPU,
    /// CPU-only deployment
    CPU { instruction_set: String },
}

impl HardwareArchitecture {
    /// Get the display name for this architecture
    pub fn display_name(&self) -> String {
        match self {
            HardwareArchitecture::AppleSilicon => "Apple Silicon".to_string(),
            HardwareArchitecture::NvidiaGPU { compute_capability } => {
                format!(
                    "NVIDIA GPU (CC {}.{})",
                    compute_capability.0, compute_capability.1
                )
            }
            HardwareArchitecture::AmdGPU { architecture } => {
                format!("AMD GPU ({})", architecture)
            }
            HardwareArchitecture::IntelGPU => "Intel GPU".to_string(),
            HardwareArchitecture::CPU { instruction_set } => {
                format!("CPU ({})", instruction_set)
            }
        }
    }

    /// Check if this architecture supports mixed precision
    pub fn supports_mixed_precision(&self) -> bool {
        match self {
            HardwareArchitecture::AppleSilicon => true,
            HardwareArchitecture::NvidiaGPU { compute_capability } => {
                // Tensor Cores available from Pascal (CC 6.0+) but really useful from Volta (CC 7.0+)
                compute_capability.0 >= 7
            }
            HardwareArchitecture::AmdGPU { architecture } => {
                // Modern AMD GPUs support mixed precision
                architecture.contains("RDNA") || architecture.contains("CDNA")
            }
            HardwareArchitecture::IntelGPU => true, // Modern Intel GPUs support FP16
            HardwareArchitecture::CPU { instruction_set } => {
                // CPU mixed precision is less beneficial but still supported
                instruction_set.contains("AVX2") || instruction_set.contains("AVX512")
            }
        }
    }

    /// Check if this architecture supports Tensor Core operations
    pub fn supports_tensor_cores(&self) -> bool {
        match self {
            HardwareArchitecture::NvidiaGPU { compute_capability } => {
                // Tensor Cores available from Volta (CC 7.0+)
                compute_capability.0 >= 7
            }
            _ => false, // Only NVIDIA GPUs have Tensor Cores
        }
    }

    /// Get the optimal batch size for this architecture
    pub fn optimal_batch_size(&self) -> usize {
        match self {
            HardwareArchitecture::AppleSilicon => 8, // Good for unified memory
            HardwareArchitecture::NvidiaGPU { compute_capability } => {
                match compute_capability {
                    (8, _) => 32, // A100, H100 - large batch sizes
                    (7, _) => 16, // V100, RTX 20/30 series
                    _ => 8,       // Older GPUs
                }
            }
            HardwareArchitecture::AmdGPU { .. } => 16,
            HardwareArchitecture::IntelGPU => 8,
            HardwareArchitecture::CPU { .. } => 1, // CPUs prefer smaller batches
        }
    }

    /// Get the estimated memory bandwidth (GB/s)
    pub fn memory_bandwidth_gbps(&self) -> f32 {
        match self {
            HardwareArchitecture::AppleSilicon => 400.0, // M2 Ultra
            HardwareArchitecture::NvidiaGPU { compute_capability } => {
                match compute_capability {
                    (9, 0) => 3350.0, // H100
                    (8, 0) => 1555.0, // A100
                    (8, 6) => 900.0,  // RTX 30 series
                    (7, 5) => 600.0,  // RTX 20 series
                    _ => 300.0,       // Older GPUs
                }
            }
            HardwareArchitecture::AmdGPU { architecture } => {
                if architecture.contains("CDNA3") {
                    5200.0 // MI300
                } else if architecture.contains("RDNA3") {
                    960.0 // RX 7900 XTX
                } else {
                    500.0 // Older AMD GPUs
                }
            }
            HardwareArchitecture::IntelGPU => 200.0, // Arc series
            HardwareArchitecture::CPU { .. } => 50.0, // Typical DDR4/5
        }
    }
}

/// Hardware capability flags
#[derive(Debug, Clone)]
pub struct HardwareCapabilities {
    /// Supports FP16 operations
    pub supports_fp16: bool,
    /// Supports BFloat16 operations
    pub supports_bfloat16: bool,
    /// Supports INT8 quantization
    pub supports_int8: bool,
    /// Supports tensor core operations
    pub supports_tensor_cores: bool,
    /// Supports SIMD instructions
    pub supports_simd: bool,
    /// Maximum SIMD width
    pub max_simd_width: usize,
    /// Supports async/concurrent execution
    pub supports_async_execution: bool,
    /// Number of compute units/cores
    pub compute_units: usize,
    /// Memory bandwidth in GB/s
    pub memory_bandwidth_gbps: f32,
}

impl HardwareCapabilities {
    /// Get capabilities for a specific hardware architecture
    pub fn for_architecture(arch: &HardwareArchitecture) -> Self {
        match arch {
            HardwareArchitecture::AppleSilicon => Self {
                supports_fp16: true,
                supports_bfloat16: false, // Conservative
                supports_int8: true,
                supports_tensor_cores: false, // Apple has AMX but different from NVIDIA Tensor Cores
                supports_simd: true,
                max_simd_width: 128, // NEON 128-bit
                supports_async_execution: true,
                compute_units: 8, // Simplified - varies by M-series chip
                memory_bandwidth_gbps: arch.memory_bandwidth_gbps(),
            },
            HardwareArchitecture::NvidiaGPU { compute_capability } => Self {
                supports_fp16: compute_capability.0 >= 6,
                supports_bfloat16: compute_capability.0 >= 8, // Ampere+
                supports_int8: compute_capability.0 >= 6,
                supports_tensor_cores: compute_capability.0 >= 7,
                supports_simd: true,
                max_simd_width: 32, // Warp size
                supports_async_execution: true,
                compute_units: match compute_capability {
                    (9, 0) => 132, // H100 SMs
                    (8, 0) => 108, // A100 SMs
                    (8, 6) => 84,  // RTX 3090 SMs
                    _ => 64,       // Estimate for older GPUs
                },
                memory_bandwidth_gbps: arch.memory_bandwidth_gbps(),
            },
            HardwareArchitecture::AmdGPU { architecture } => Self {
                supports_fp16: true,
                supports_bfloat16: architecture.contains("CDNA3"),
                supports_int8: true,
                supports_tensor_cores: false, // AMD has Matrix Cores but different
                supports_simd: true,
                max_simd_width: 64, // Wavefront size
                supports_async_execution: true,
                compute_units: if architecture.contains("CDNA3") {
                    304 // MI300
                } else if architecture.contains("RDNA3") {
                    96 // RX 7900 XTX
                } else {
                    64 // Estimate
                },
                memory_bandwidth_gbps: arch.memory_bandwidth_gbps(),
            },
            HardwareArchitecture::IntelGPU => Self {
                supports_fp16: true,
                supports_bfloat16: true, // Arc series supports BF16
                supports_int8: true,
                supports_tensor_cores: false,
                supports_simd: true,
                max_simd_width: 16, // SIMD16
                supports_async_execution: true,
                compute_units: 32, // Arc A770 Xe cores
                memory_bandwidth_gbps: arch.memory_bandwidth_gbps(),
            },
            HardwareArchitecture::CPU { instruction_set } => Self {
                supports_fp16: false, // Native FP16 not common on x86
                supports_bfloat16: instruction_set.contains("AVX512_BF16"),
                supports_int8: true,
                supports_tensor_cores: false,
                supports_simd: instruction_set.contains("AVX") || instruction_set.contains("SSE"),
                max_simd_width: if instruction_set.contains("AVX512") {
                    512
                } else if instruction_set.contains("AVX2") {
                    256
                } else if instruction_set.contains("AVX") {
                    256
                } else {
                    128 // SSE
                },
                supports_async_execution: true,
                compute_units: 8, // Estimate for modern CPUs
                memory_bandwidth_gbps: arch.memory_bandwidth_gbps(),
            },
        }
    }
}

/// Hardware-specific optimization configurations
pub struct HardwareOptimizationConfigs;

impl HardwareOptimizationConfigs {
    /// Create TensorRT-style optimization configuration for maximum performance.
    pub fn tensorrt_optimization_config() -> OptimizationConfig {
        OptimizationConfig {
            constant_folding: true,
            dead_code_elimination: true,
            redundant_ops_removal: true,
            batch_norm_folding: true,
            target_precision: Some(DType::Float16), // Mixed precision for speed
            max_memory: Some(2 * 1024 * 1024 * 1024), // 2GB limit
            kernel_fusion: true,
            memory_layout_optimization: true,
            dynamic_batching: false, // Fixed batch for maximum optimization
            cuda_graph_optimization: true, // Enable CUDA graphs if available
            optimization_level: 2,   // Maximum optimization
            target_batch_size: Some(1), // Single inference optimization
            quantization_aware: false,
        }
    }

    /// Create TensorRT-style optimization configuration for inference servers.
    pub fn inference_server_optimization_config() -> OptimizationConfig {
        OptimizationConfig {
            constant_folding: true,
            dead_code_elimination: true,
            redundant_ops_removal: true,
            batch_norm_folding: true,
            target_precision: Some(DType::Float16),
            max_memory: Some(4 * 1024 * 1024 * 1024), // 4GB for server
            kernel_fusion: true,
            memory_layout_optimization: true,
            dynamic_batching: true, // Enable dynamic batching for servers
            cuda_graph_optimization: true,
            optimization_level: 2,
            target_batch_size: Some(8), // Optimized for batch inference
            quantization_aware: false,
        }
    }

    /// Create optimization configuration for edge deployment.
    pub fn edge_deployment_optimization_config() -> OptimizationConfig {
        OptimizationConfig {
            constant_folding: true,
            dead_code_elimination: true,
            redundant_ops_removal: true,
            batch_norm_folding: true,
            target_precision: Some(DType::Int8), // Quantized for edge
            max_memory: Some(512 * 1024 * 1024), // 512MB for edge devices
            kernel_fusion: true,
            memory_layout_optimization: true,
            dynamic_batching: false,        // Fixed batch for edge
            cuda_graph_optimization: false, // Not available on most edge devices
            optimization_level: 1,          // Balanced optimization
            target_batch_size: Some(1),     // Single inference for edge
            quantization_aware: true,       // Enable quantization-aware optimizations
        }
    }

    /// Create optimization configuration for mobile deployment.
    pub fn mobile_deployment_optimization_config() -> OptimizationConfig {
        OptimizationConfig {
            constant_folding: true,
            dead_code_elimination: true,
            redundant_ops_removal: true,
            batch_norm_folding: true,
            target_precision: Some(DType::Float16), // F16 for mobile GPUs
            max_memory: Some(256 * 1024 * 1024),    // 256MB for mobile
            kernel_fusion: true,
            memory_layout_optimization: true,
            dynamic_batching: false,        // Fixed batch for mobile
            cuda_graph_optimization: false, // Not applicable for mobile
            optimization_level: 1,          // Balanced for battery life
            target_batch_size: Some(1),     // Single inference for mobile
            quantization_aware: true,
        }
    }

    /// Create conservative optimization configuration.
    pub fn conservative_optimization_config() -> OptimizationConfig {
        OptimizationConfig {
            constant_folding: true,
            dead_code_elimination: false, // Conservative: don't remove code
            redundant_ops_removal: true,
            batch_norm_folding: true,
            target_precision: None, // Keep original precision
            max_memory: None,       // No memory limits
            kernel_fusion: true,
            memory_layout_optimization: false, // Conservative: keep original layouts
            dynamic_batching: false,
            cuda_graph_optimization: false,
            optimization_level: 0, // Basic optimization only
            target_batch_size: Some(1),
            quantization_aware: false,
        }
    }

    /// Create Apple Silicon optimized configuration.
    pub fn apple_silicon_optimization_config() -> OptimizationConfig {
        OptimizationConfig {
            constant_folding: true,
            dead_code_elimination: true,
            redundant_ops_removal: true,
            batch_norm_folding: true,
            target_precision: Some(DType::Float16), // Apple Silicon supports FP16
            max_memory: Some(1024 * 1024 * 1024),   // 1GB for unified memory
            kernel_fusion: true,
            memory_layout_optimization: true,
            dynamic_batching: false,
            cuda_graph_optimization: false, // Not applicable
            optimization_level: 2,          // Maximum for Apple Silicon
            target_batch_size: Some(8),     // Good for unified memory
            quantization_aware: true,
        }
    }

    /// Create CPU-optimized configuration.
    pub fn cpu_optimization_config(instruction_set: &str) -> OptimizationConfig {
        OptimizationConfig {
            constant_folding: true,
            dead_code_elimination: true,
            redundant_ops_removal: true,
            batch_norm_folding: true,
            target_precision: if instruction_set.contains("AVX512_BF16") {
                Some(DType::Float16) // Use BF16 if available
            } else {
                None // Stick to FP32 for CPU
            },
            max_memory: Some(8 * 1024 * 1024 * 1024), // 8GB for CPU
            kernel_fusion: true,
            memory_layout_optimization: true,
            dynamic_batching: false,
            cuda_graph_optimization: false,
            optimization_level: 1,      // Balanced for CPU
            target_batch_size: Some(1), // CPUs prefer smaller batches
            quantization_aware: true,   // INT8 beneficial for CPU
        }
    }

    /// Get optimization config for specific hardware architecture.
    pub fn for_architecture(arch: &HardwareArchitecture) -> OptimizationConfig {
        match arch {
            HardwareArchitecture::AppleSilicon => Self::apple_silicon_optimization_config(),
            HardwareArchitecture::NvidiaGPU { compute_capability } => {
                if compute_capability.0 >= 8 {
                    Self::tensorrt_optimization_config() // Modern GPUs
                } else {
                    Self::conservative_optimization_config() // Older GPUs
                }
            }
            HardwareArchitecture::AmdGPU { architecture } => {
                if architecture.contains("RDNA3") || architecture.contains("CDNA") {
                    Self::tensorrt_optimization_config() // Modern AMD
                } else {
                    Self::conservative_optimization_config() // Older AMD
                }
            }
            HardwareArchitecture::IntelGPU => Self::edge_deployment_optimization_config(),
            HardwareArchitecture::CPU { instruction_set } => {
                Self::cpu_optimization_config(instruction_set)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hardware_architecture_display() {
        let apple = HardwareArchitecture::AppleSilicon;
        assert_eq!(apple.display_name(), "Apple Silicon");

        let nvidia = HardwareArchitecture::NvidiaGPU {
            compute_capability: (8, 0),
        };
        assert_eq!(nvidia.display_name(), "NVIDIA GPU (CC 8.0)");

        let amd = HardwareArchitecture::AmdGPU {
            architecture: "RDNA3".to_string(),
        };
        assert_eq!(amd.display_name(), "AMD GPU (RDNA3)");
    }

    #[test]
    fn test_mixed_precision_support() {
        let apple = HardwareArchitecture::AppleSilicon;
        assert!(apple.supports_mixed_precision());

        let nvidia_modern = HardwareArchitecture::NvidiaGPU {
            compute_capability: (8, 0),
        };
        assert!(nvidia_modern.supports_mixed_precision());

        let nvidia_old = HardwareArchitecture::NvidiaGPU {
            compute_capability: (6, 0),
        };
        assert!(!nvidia_old.supports_mixed_precision());

        let cpu = HardwareArchitecture::CPU {
            instruction_set: "AVX2".to_string(),
        };
        assert!(cpu.supports_mixed_precision());
    }

    #[test]
    fn test_tensor_core_support() {
        let nvidia_volta = HardwareArchitecture::NvidiaGPU {
            compute_capability: (7, 0),
        };
        assert!(nvidia_volta.supports_tensor_cores());

        let nvidia_pascal = HardwareArchitecture::NvidiaGPU {
            compute_capability: (6, 1),
        };
        assert!(!nvidia_pascal.supports_tensor_cores());

        let apple = HardwareArchitecture::AppleSilicon;
        assert!(!apple.supports_tensor_cores()); // Only NVIDIA has Tensor Cores
    }

    #[test]
    fn test_optimal_batch_size() {
        let apple = HardwareArchitecture::AppleSilicon;
        assert_eq!(apple.optimal_batch_size(), 8);

        let a100 = HardwareArchitecture::NvidiaGPU {
            compute_capability: (8, 0),
        };
        assert_eq!(a100.optimal_batch_size(), 32);

        let cpu = HardwareArchitecture::CPU {
            instruction_set: "AVX2".to_string(),
        };
        assert_eq!(cpu.optimal_batch_size(), 1);
    }

    #[test]
    fn test_memory_bandwidth() {
        let a100 = HardwareArchitecture::NvidiaGPU {
            compute_capability: (8, 0),
        };
        assert_eq!(a100.memory_bandwidth_gbps(), 1555.0);

        let h100 = HardwareArchitecture::NvidiaGPU {
            compute_capability: (9, 0),
        };
        assert_eq!(h100.memory_bandwidth_gbps(), 3350.0);
    }

    #[test]
    fn test_hardware_capabilities() {
        let nvidia_ampere = HardwareArchitecture::NvidiaGPU {
            compute_capability: (8, 6),
        };
        let caps = HardwareCapabilities::for_architecture(&nvidia_ampere);

        assert!(caps.supports_fp16);
        assert!(caps.supports_bfloat16);
        assert!(caps.supports_int8);
        assert!(caps.supports_tensor_cores);
        assert!(caps.supports_simd);
        assert_eq!(caps.max_simd_width, 32);
    }

    #[test]
    fn test_optimization_configs() {
        let tensorrt_config = HardwareOptimizationConfigs::tensorrt_optimization_config();
        assert!(tensorrt_config.kernel_fusion);
        assert!(tensorrt_config.memory_layout_optimization);
        assert_eq!(tensorrt_config.optimization_level, 2);

        let edge_config = HardwareOptimizationConfigs::edge_deployment_optimization_config();
        assert_eq!(edge_config.target_precision, Some(DType::Int8));
        assert!(edge_config.quantization_aware);
        assert!(!edge_config.dynamic_batching);

        let mobile_config = HardwareOptimizationConfigs::mobile_deployment_optimization_config();
        assert_eq!(mobile_config.target_precision, Some(DType::Float16));
        assert_eq!(mobile_config.max_memory, Some(256 * 1024 * 1024));
    }

    #[test]
    fn test_architecture_specific_configs() {
        let apple = HardwareArchitecture::AppleSilicon;
        let apple_config = HardwareOptimizationConfigs::for_architecture(&apple);
        assert_eq!(apple_config.target_batch_size, Some(8));

        let nvidia_modern = HardwareArchitecture::NvidiaGPU {
            compute_capability: (8, 0),
        };
        let nvidia_config = HardwareOptimizationConfigs::for_architecture(&nvidia_modern);
        assert_eq!(nvidia_config.optimization_level, 2);

        let cpu = HardwareArchitecture::CPU {
            instruction_set: "AVX2".to_string(),
        };
        let cpu_config = HardwareOptimizationConfigs::for_architecture(&cpu);
        assert_eq!(cpu_config.target_batch_size, Some(1));
    }
}
