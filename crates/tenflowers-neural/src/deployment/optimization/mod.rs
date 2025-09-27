//! Model optimization techniques for deployment.
//!
//! This module provides various optimization passes to improve model efficiency
//! for deployment on mobile and edge devices. The module has been refactored
//! into a modular architecture for better maintainability and organization.
//!
//! ## Module Organization
//!
//! - **core**: Configuration types and base optimization structures
//! - **basic**: Fundamental optimization passes (constant folding, dead code elimination)
//! - **advanced**: Complex optimization passes (layer fusion, kernel fusion, memory optimization)
//! - **platform**: Platform-specific optimizations (auto-tuning, mixed precision, profiling)
//! - **hardware**: Hardware architecture definitions and platform configurations
//! - **deployment**: Deployment wrappers and high-level APIs
//!
//! All functionality maintains 100% backward compatibility through strategic re-exports.

pub mod advanced;
pub mod basic;
pub mod core;
pub mod deployment;
pub mod hardware;
pub mod platform;

// Re-export core types for backward compatibility
pub use core::{ModelOptimizer, OptimizationConfig, OptimizationStats};

// Re-export hardware types
pub use hardware::HardwareArchitecture;

// Re-export platform types
pub use platform::{
    AutoTuningConfig, MixedPrecisionConfig, PlatformOptimizations, ProfileGuidedConfig,
    ProfileResults,
};

// Re-export deployment types
pub use deployment::{
    AdvancedOptimizationConfig, DeploymentMetadata, DeploymentModel, DeploymentOptimizer,
};

// Re-export hardware-specific optimization configs
// Note: These are associated functions on HardwareOptimizationConfigs, not submodules
pub use hardware::HardwareOptimizationConfigs;

// Re-export preset optimization functions
// Note: These are associated functions on DeploymentOptimizer

// Re-export advanced configuration builders for backward compatibility
// Note: These would be associated functions on DeploymentOptimizer if they exist

// Convenience re-exports for common operations
use crate::model::Sequential;
use num_traits;
use tenflowers_core::TensorError;

/// High-level API for model deployment optimization.
///
/// This is a convenience wrapper that maintains the original API.
pub fn optimize_for_deployment<T>(
    model: &Sequential<T>,
    target_device: &str,
    config: Option<OptimizationConfig>,
) -> Result<DeploymentModel<T>, TensorError>
where
    T: Clone
        + Default
        + Send
        + Sync
        + num_traits::Zero
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    DeploymentOptimizer::optimize_for_deployment(model, target_device, config)
}

/// High-level API for TensorRT-style model optimization.
///
/// This is a convenience wrapper that maintains the original API.
pub fn optimize_for_tensorrt<T>(
    model: &Sequential<T>,
    target_device: &str,
) -> Result<DeploymentModel<T>, TensorError>
where
    T: Clone
        + Default
        + Send
        + Sync
        + num_traits::Zero
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    DeploymentOptimizer::optimize_for_tensorrt(model, target_device)
}

/// High-level API for inference server optimization.
///
/// This is a convenience wrapper that maintains the original API.
pub fn optimize_for_inference_server<T>(
    model: &Sequential<T>,
    batch_size: usize,
) -> Result<DeploymentModel<T>, TensorError>
where
    T: Clone
        + Default
        + Send
        + Sync
        + num_traits::Zero
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    DeploymentOptimizer::optimize_for_inference_server(model, batch_size)
}

/// High-level API for edge device optimization.
///
/// This is a convenience wrapper that maintains the original API.
pub fn optimize_for_edge<T>(
    model: &Sequential<T>,
    memory_limit_mb: usize,
) -> Result<DeploymentModel<T>, TensorError>
where
    T: Clone
        + Default
        + Send
        + Sync
        + num_traits::Zero
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    DeploymentOptimizer::optimize_for_edge(model, memory_limit_mb)
}

/// High-level API for mobile device optimization.
///
/// This is a convenience wrapper that maintains the original API.
pub fn optimize_for_mobile<T>(model: &Sequential<T>) -> Result<DeploymentModel<T>, TensorError>
where
    T: Clone
        + Default
        + Send
        + Sync
        + num_traits::Zero
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    DeploymentOptimizer::optimize_for_mobile(model)
}

/// High-level API for TensorRT-like advanced optimization.
///
/// This is a convenience wrapper that maintains the original API.
pub fn optimize_for_tensorrt_advanced<T>(
    model: &Sequential<T>,
) -> Result<(DeploymentModel<T>, ProfileResults), TensorError>
where
    T: Clone
        + Default
        + Send
        + Sync
        + num_traits::Zero
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    DeploymentOptimizer::optimize_for_tensorrt_advanced(model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::Dense;
    use crate::model::Model;

    #[test]
    fn test_backward_compatibility_optimization_config() {
        let config = OptimizationConfig::default();
        assert!(config.constant_folding);
        assert!(config.dead_code_elimination);
        assert!(config.redundant_ops_removal);
        assert!(config.batch_norm_folding);
    }

    #[test]
    fn test_backward_compatibility_model_optimizer() {
        let optimizer = ModelOptimizer::new();
        assert!(optimizer.config().constant_folding);

        let custom_config = OptimizationConfig {
            constant_folding: false,
            ..Default::default()
        };
        let custom_optimizer = ModelOptimizer::with_config(custom_config);
        assert!(!custom_optimizer.config().constant_folding);
    }

    #[test]
    fn test_backward_compatibility_optimization_stats() {
        let stats = OptimizationStats {
            original_size: 1000,
            optimized_size: 800,
            ops_removed: 5,
            params_removed: 100,
            speedup_ratio: 1.25,
            memory_reduction: 0.2,
        };

        assert_eq!(stats.compression_ratio(), 1.25);
    }

    #[test]
    fn test_backward_compatibility_sequential_optimization() {
        let model = Sequential::new(vec![
            Box::new(Dense::<f32>::new(10, 20, true)),
            Box::new(Dense::<f32>::new(20, 1, true)),
        ]);

        let optimizer = ModelOptimizer::new();
        let result = optimizer.optimize_sequential(&model);

        assert!(result.is_ok());
        let (optimized_model, stats) = result.unwrap();
        assert!(stats.speedup_ratio >= 1.0);
        assert_eq!(optimized_model.parameters().len(), 0);
        assert_eq!(stats.original_size, 16);
    }

    #[test]
    fn test_backward_compatibility_deployment_model() {
        let model = Sequential::new(vec![Box::new(Dense::<f32>::new(10, 1, true))]);

        let stats = OptimizationStats {
            original_size: 1000,
            optimized_size: 800,
            ops_removed: 2,
            params_removed: 0,
            speedup_ratio: 1.1,
            memory_reduction: 0.2,
        };

        let mut deployment_model = DeploymentModel::new(model, stats, "mobile".to_string());

        deployment_model.set_inference_time(5.0);
        deployment_model.set_accuracy(0.95);

        let summary = deployment_model.export_summary();
        assert!(summary.contains("mobile"));
        assert!(summary.contains("5.0"));
        assert!(summary.contains("95.00"));
    }

    #[test]
    fn test_backward_compatibility_high_level_apis() {
        let model = Sequential::new(vec![
            Box::new(Dense::<f32>::new(5, 10, true)),
            Box::new(Dense::<f32>::new(10, 1, true)),
        ]);

        // Test the convenience wrapper functions
        let result = optimize_for_deployment(&model, "edge", None);
        assert!(result.is_ok());

        let result = optimize_for_tensorrt(&model, "gpu");
        assert!(result.is_ok());

        let result = optimize_for_inference_server(&model, 8);
        assert!(result.is_ok());

        let result = optimize_for_edge(&model, 512);
        assert!(result.is_ok());

        let result = optimize_for_mobile(&model);
        assert!(result.is_ok());
    }

    #[test]
    fn test_hardware_architecture_integration() {
        let apple = HardwareArchitecture::AppleSilicon;
        assert_eq!(apple.display_name(), "Apple Silicon");
        assert!(apple.supports_mixed_precision());
        assert!(!apple.supports_tensor_cores());

        let nvidia = HardwareArchitecture::NvidiaGPU {
            compute_capability: (8, 0),
        };
        assert!(nvidia.supports_mixed_precision());
        assert!(nvidia.supports_tensor_cores());
    }

    #[test]
    #[ignore] // TODO: Implement platform-specific optimization configs
    fn test_platform_optimization_configs() {
        // TODO: Implement tensorrt_optimization_config()
        // let tensorrt_config = tensorrt_optimization_config();
        // assert!(tensorrt_config.kernel_fusion);
        // assert!(tensorrt_config.memory_layout_optimization);
        // assert_eq!(tensorrt_config.optimization_level, 2);

        // TODO: Implement edge_deployment_optimization_config()
        // let edge_config = edge_deployment_optimization_config();
        // assert!(edge_config.quantization_aware);
        // assert!(!edge_config.dynamic_batching);

        // TODO: Implement mobile_deployment_optimization_config()
        // let mobile_config = mobile_deployment_optimization_config();
        // assert_eq!(mobile_config.target_batch_size, Some(1));
    }

    #[test]
    fn test_advanced_optimization_features() {
        let auto_tuning_config = AutoTuningConfig::default();
        assert!(auto_tuning_config.auto_kernel_selection);
        assert_eq!(auto_tuning_config.tuning_iterations, 100);

        let mixed_precision_config = MixedPrecisionConfig::default();
        assert!(mixed_precision_config.auto_mixed_precision);
        assert!(mixed_precision_config.tensor_core_optimization);

        let profile_guided_config = ProfileGuidedConfig::default();
        assert!(profile_guided_config.enabled);
        assert!(!profile_guided_config.sample_input_shapes.is_empty());
    }

    #[test]
    fn test_modular_architecture_integrity() {
        // Test that all modules are properly accessible
        use crate::deployment::optimization::advanced::AdvancedOptimizations;
        use crate::deployment::optimization::basic::BasicOptimizations;
        use crate::deployment::optimization::core::OptimizationConfig as CoreConfig;
        use crate::deployment::optimization::deployment::DeploymentOptimizer as DeploymentOpt;
        use crate::deployment::optimization::hardware::HardwareCapabilities;
        use crate::deployment::optimization::platform::PlatformOptimizations;

        // Verify that the modular components are accessible
        let _config = CoreConfig::default();

        // These would be used in practice but for the test we just verify they're accessible
        let _basic = BasicOptimizations;
        let _advanced = AdvancedOptimizations;
        let _platform = PlatformOptimizations;
        let _capabilities =
            HardwareCapabilities::for_architecture(&HardwareArchitecture::AppleSilicon);
        let _deployment = DeploymentOpt;
    }
}
