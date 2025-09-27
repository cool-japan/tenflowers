//! Deployment wrappers and high-level APIs
//!
//! This module provides deployment-ready model wrappers, metadata management,
//! and high-level optimization APIs for different deployment scenarios.

use crate::model::Sequential;
use num_traits;
#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tenflowers_core::TensorError;

use super::core::{ModelOptimizer, OptimizationConfig, OptimizationStats};
use super::hardware::{HardwareArchitecture, HardwareOptimizationConfigs};
use super::platform::{
    AutoTuningConfig, MixedPrecisionConfig, ProfileGuidedConfig, ProfileResults,
};

/// Deployment-ready model wrapper.
pub struct DeploymentModel<T>
where
    T: Clone,
{
    model: Sequential<T>,
    optimization_stats: OptimizationStats,
    metadata: DeploymentMetadata,
}

/// Metadata for deployment models.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct DeploymentMetadata {
    /// Target device type
    pub target_device: String,
    /// Optimization level applied
    pub optimization_level: String,
    /// Estimated inference time (ms)
    pub estimated_inference_time: Option<f32>,
    /// Model accuracy on validation set
    pub accuracy: Option<f32>,
    /// Creation timestamp
    #[cfg(feature = "serialize")]
    pub created_at: chrono::DateTime<chrono::Utc>,
    #[cfg(not(feature = "serialize"))]
    pub created_at: String,
}

impl<T> DeploymentModel<T>
where
    T: Clone + Default + 'static,
{
    /// Create a new deployment model.
    pub fn new(
        model: Sequential<T>,
        optimization_stats: OptimizationStats,
        target_device: String,
    ) -> Self {
        let metadata = DeploymentMetadata {
            target_device,
            optimization_level: "standard".to_string(),
            estimated_inference_time: None,
            accuracy: None,
            #[cfg(feature = "serialize")]
            created_at: chrono::Utc::now(),
            #[cfg(not(feature = "serialize"))]
            created_at: "N/A".to_string(),
        };

        Self {
            model,
            optimization_stats,
            metadata,
        }
    }

    /// Get the underlying model.
    pub fn model(&self) -> &Sequential<T> {
        &self.model
    }

    /// Get optimization statistics.
    pub fn stats(&self) -> &OptimizationStats {
        &self.optimization_stats
    }

    /// Get deployment metadata.
    pub fn metadata(&self) -> &DeploymentMetadata {
        &self.metadata
    }

    /// Update estimated inference time.
    pub fn set_inference_time(&mut self, time_ms: f32) {
        self.metadata.estimated_inference_time = Some(time_ms);
    }

    /// Update model accuracy.
    pub fn set_accuracy(&mut self, accuracy: f32) {
        self.metadata.accuracy = Some(accuracy);
    }

    /// Export model summary for deployment.
    pub fn export_summary(&self) -> String {
        format!(
            "Deployment Model Summary
Target Device: {}
Optimization Level: {}
Original Size: {} bytes
Optimized Size: {} bytes
Compression Ratio: {:.2}x
Memory Reduction: {:.1}%
Estimated Speedup: {:.2}x
Operations Removed: {}
Parameters Removed: {}
Estimated Inference Time: {} ms
Accuracy: {}%
Created: {}",
            self.metadata.target_device,
            self.metadata.optimization_level,
            self.optimization_stats.original_size,
            self.optimization_stats.optimized_size,
            self.optimization_stats.compression_ratio(),
            self.optimization_stats.memory_reduction * 100.0,
            self.optimization_stats.speedup_ratio,
            self.optimization_stats.ops_removed,
            self.optimization_stats.params_removed,
            self.metadata
                .estimated_inference_time
                .map_or("N/A".to_string(), |t| t.to_string()),
            self.metadata
                .accuracy
                .map_or("N/A".to_string(), |a| format!("{:.2}", a * 100.0)),
            {
                #[cfg(feature = "serialize")]
                {
                    self.metadata
                        .created_at
                        .format("%Y-%m-%d %H:%M:%S UTC")
                        .to_string()
                }
                #[cfg(not(feature = "serialize"))]
                {
                    self.metadata.created_at.clone()
                }
            }
        )
    }
}

/// Enhanced optimization configuration with TensorRT-like advanced features.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct AdvancedOptimizationConfig {
    /// Base optimization configuration
    pub base_config: OptimizationConfig,
    /// Auto-tuning configuration
    pub auto_tuning: AutoTuningConfig,
    /// Profile-guided optimization
    pub profile_guided: ProfileGuidedConfig,
    /// Advanced mixed precision settings
    pub mixed_precision: MixedPrecisionConfig,
    /// Enable calibration dataset validation
    pub enable_accuracy_validation: bool,
    /// Calibration dataset size for accuracy validation
    pub calibration_dataset_size: usize,
    /// Maximum acceptable accuracy drop
    pub max_accuracy_drop: f32,
    /// Enable sparsity-aware optimization
    pub sparsity_aware: bool,
    /// Target inference latency in milliseconds
    pub target_latency_ms: Option<f32>,
    /// Target memory usage in bytes
    pub target_memory_bytes: Option<usize>,
}

impl Default for AdvancedOptimizationConfig {
    fn default() -> Self {
        Self {
            base_config: OptimizationConfig::default(),
            auto_tuning: AutoTuningConfig::default(),
            profile_guided: ProfileGuidedConfig::default(),
            mixed_precision: MixedPrecisionConfig::default(),
            enable_accuracy_validation: true,
            calibration_dataset_size: 1000,
            max_accuracy_drop: 0.01, // 1% maximum drop
            sparsity_aware: true,
            target_latency_ms: None,
            target_memory_bytes: None,
        }
    }
}

/// High-level deployment optimization API
pub struct DeploymentOptimizer;

impl DeploymentOptimizer {
    /// High-level API for model deployment optimization.
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
        let optimizer = ModelOptimizer::with_config(config.unwrap_or_default());
        let (optimized_model, stats) = optimizer.optimize_sequential(model)?;

        Ok(DeploymentModel::new(
            optimized_model,
            stats,
            target_device.to_string(),
        ))
    }

    /// High-level API for TensorRT-style model optimization.
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
        let config = HardwareOptimizationConfigs::tensorrt_optimization_config();
        Self::optimize_for_deployment(model, target_device, Some(config))
    }

    /// High-level API for inference server optimization.
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
        let mut config = HardwareOptimizationConfigs::inference_server_optimization_config();
        config.target_batch_size = Some(batch_size);
        Self::optimize_for_deployment(model, "inference_server", Some(config))
    }

    /// High-level API for edge device optimization.
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
        let mut config = HardwareOptimizationConfigs::edge_deployment_optimization_config();
        config.max_memory = Some(memory_limit_mb * 1024 * 1024);
        Self::optimize_for_deployment(model, "edge_device", Some(config))
    }

    /// High-level API for mobile device optimization.
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
        let config = HardwareOptimizationConfigs::mobile_deployment_optimization_config();
        Self::optimize_for_deployment(model, "mobile_device", Some(config))
    }

    /// High-level API for TensorRT-like advanced optimization.
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
        let config = Self::tensorrt_advanced_optimization_config();
        let optimizer = ModelOptimizer::with_config(config.base_config);
        let (optimized_model, stats) = optimizer.optimize_sequential(model)?;

        let deployment_model = DeploymentModel::new(optimized_model, stats, "TensorRT".to_string());

        // Create basic profile results
        let profile_results = super::platform::ProfileResults {
            avg_inference_time_ms: 0.0,
            peak_memory_bytes: 0,
            memory_bandwidth_utilization: 0.0,
            compute_utilization: 0.0,
            kernel_times: HashMap::new(),
            batch_throughput: HashMap::new(),
            accuracy_by_precision: HashMap::new(),
        };

        Ok((deployment_model, profile_results))
    }

    /// Create TensorRT-like optimization configuration for maximum performance.
    pub fn tensorrt_advanced_optimization_config() -> AdvancedOptimizationConfig {
        AdvancedOptimizationConfig {
            base_config: HardwareOptimizationConfigs::tensorrt_optimization_config(),
            auto_tuning: AutoTuningConfig {
                auto_kernel_selection: true,
                tuning_iterations: 200,
                profile_time_ms: 2000,
                hardware_specific: true,
                target_architecture: HardwareArchitecture::NvidiaGPU {
                    compute_capability: (8, 0), // A100
                },
                performance_tolerance: 0.02, // 2% tolerance
                mixed_precision_tuning: true,
            },
            profile_guided: ProfileGuidedConfig {
                enabled: true,
                warmup_iterations: 20,
                profile_iterations: 200,
                sample_input_shapes: vec![
                    vec![1, 224, 224, 3],
                    vec![8, 224, 224, 3],
                    vec![16, 224, 224, 3],
                    vec![32, 224, 224, 3],
                ],
                profile_memory: true,
                profile_kernels: true,
                profile_batch_sizes: vec![1, 8, 16, 32, 64],
            },
            mixed_precision: MixedPrecisionConfig {
                auto_mixed_precision: true,
                loss_scaling: 4096.0,
                tensor_core_optimization: true,
                enable_bfloat16: true,
                dynamic_loss_scaling: true,
                ..Default::default()
            },
            enable_accuracy_validation: true,
            calibration_dataset_size: 2000,
            max_accuracy_drop: 0.005, // 0.5% maximum drop
            sparsity_aware: true,
            target_latency_ms: Some(10.0), // 10ms target
            target_memory_bytes: Some(4 * 1024 * 1024 * 1024), // 4GB
        }
    }

    /// Create Apple Silicon optimized configuration.
    pub fn apple_silicon_optimization_config() -> AdvancedOptimizationConfig {
        AdvancedOptimizationConfig {
            base_config: HardwareOptimizationConfigs::apple_silicon_optimization_config(),
            auto_tuning: AutoTuningConfig {
                target_architecture: HardwareArchitecture::AppleSilicon,
                tuning_iterations: 150,
                hardware_specific: true,
                ..Default::default()
            },
            mixed_precision: MixedPrecisionConfig {
                auto_mixed_precision: true,
                tensor_core_optimization: false, // Apple Silicon doesn't have Tensor Cores
                enable_bfloat16: false,          // Conservative for Apple Silicon
                ..Default::default()
            },
            target_latency_ms: Some(5.0), // 5ms target for mobile
            target_memory_bytes: Some(1024 * 1024 * 1024), // 1GB for mobile
            ..Default::default()
        }
    }

    /// Optimize for specific hardware architecture.
    pub fn optimize_for_architecture<T>(
        model: &Sequential<T>,
        architecture: &HardwareArchitecture,
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
        let config = HardwareOptimizationConfigs::for_architecture(architecture);
        let target_device = architecture.display_name();
        Self::optimize_for_deployment(model, &target_device, Some(config))
    }
}

/// Convenience functions for common optimization scenarios
pub mod presets {
    use super::*;

    /// Optimize model for maximum performance (GPU inference)
    pub fn optimize_for_maximum_performance<T>(
        model: &Sequential<T>,
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
        DeploymentOptimizer::optimize_for_tensorrt(model, "gpu_inference")
    }

    /// Optimize model for minimal memory usage (edge devices)
    pub fn optimize_for_minimal_memory<T>(
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

    /// Optimize model for balanced performance/efficiency (mobile)
    pub fn optimize_for_balanced<T>(
        model: &Sequential<T>,
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
        DeploymentOptimizer::optimize_for_mobile(model)
    }

    /// Optimize model for high-throughput inference (servers)
    pub fn optimize_for_throughput<T>(
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

    /// Optimize model conservatively (minimal risk)
    pub fn optimize_conservatively<T>(
        model: &Sequential<T>,
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
        let config = HardwareOptimizationConfigs::conservative_optimization_config();
        DeploymentOptimizer::optimize_for_deployment(model, "conservative", Some(config))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::Dense;

    #[test]
    fn test_deployment_model_creation() {
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
    fn test_optimize_for_deployment() {
        let model = Sequential::new(vec![
            Box::new(Dense::<f32>::new(5, 10, true)),
            Box::new(Dense::<f32>::new(10, 1, true)),
        ]);

        let result = DeploymentOptimizer::optimize_for_deployment(&model, "edge", None);
        assert!(result.is_ok());

        let deployment_model = result.unwrap();
        assert_eq!(deployment_model.metadata().target_device, "edge");
        assert!(deployment_model.stats().speedup_ratio >= 1.0);
    }

    #[test]
    fn test_optimize_for_tensorrt() {
        let model = Sequential::new(vec![
            Box::new(Dense::<f32>::new(8, 16, true)),
            Box::new(Dense::<f32>::new(16, 8, true)),
        ]);

        let result = DeploymentOptimizer::optimize_for_tensorrt(&model, "gpu");
        assert!(result.is_ok());

        let deployment_model = result.unwrap();
        assert_eq!(deployment_model.metadata().target_device, "gpu");
    }

    #[test]
    fn test_optimize_for_mobile() {
        let model = Sequential::new(vec![Box::new(Dense::<f32>::new(4, 8, true))]);

        let result = DeploymentOptimizer::optimize_for_mobile(&model);
        assert!(result.is_ok());

        let deployment_model = result.unwrap();
        assert_eq!(deployment_model.metadata().target_device, "mobile_device");
    }

    #[test]
    fn test_optimize_for_edge() {
        let model = Sequential::new(vec![Box::new(Dense::<f32>::new(3, 6, true))]);

        let result = DeploymentOptimizer::optimize_for_edge(&model, 512);
        assert!(result.is_ok());

        let deployment_model = result.unwrap();
        assert_eq!(deployment_model.metadata().target_device, "edge_device");
    }

    #[test]
    fn test_optimize_for_inference_server() {
        let model = Sequential::new(vec![Box::new(Dense::<f32>::new(6, 12, true))]);

        let result = DeploymentOptimizer::optimize_for_inference_server(&model, 16);
        assert!(result.is_ok());

        let deployment_model = result.unwrap();
        assert_eq!(
            deployment_model.metadata().target_device,
            "inference_server"
        );
    }

    #[test]
    fn test_optimize_for_architecture() {
        let model = Sequential::new(vec![Box::new(Dense::<f32>::new(2, 4, true))]);

        let architecture = HardwareArchitecture::AppleSilicon;
        let result = DeploymentOptimizer::optimize_for_architecture(&model, &architecture);
        assert!(result.is_ok());

        let deployment_model = result.unwrap();
        assert_eq!(deployment_model.metadata().target_device, "Apple Silicon");
    }

    #[test]
    fn test_advanced_optimization_config() {
        let config = DeploymentOptimizer::tensorrt_advanced_optimization_config();
        assert!(config.base_config.kernel_fusion);
        assert!(config.auto_tuning.auto_kernel_selection);
        assert!(config.profile_guided.enabled);
        assert!(config.mixed_precision.auto_mixed_precision);
        assert!(config.enable_accuracy_validation);
    }

    #[test]
    fn test_apple_silicon_config() {
        let config = DeploymentOptimizer::apple_silicon_optimization_config();
        assert!(!config.mixed_precision.tensor_core_optimization); // Apple doesn't have Tensor Cores
        assert!(!config.mixed_precision.enable_bfloat16); // Conservative
        assert_eq!(config.target_latency_ms, Some(5.0));
    }

    #[test]
    fn test_preset_functions() {
        let model = Sequential::new(vec![Box::new(Dense::<f32>::new(2, 2, true))]);

        // Test maximum performance preset
        let result = presets::optimize_for_maximum_performance(&model);
        assert!(result.is_ok());

        // Test minimal memory preset
        let result = presets::optimize_for_minimal_memory(&model, 256);
        assert!(result.is_ok());

        // Test balanced preset
        let result = presets::optimize_for_balanced(&model);
        assert!(result.is_ok());

        // Test throughput preset
        let result = presets::optimize_for_throughput(&model, 8);
        assert!(result.is_ok());

        // Test conservative preset
        let result = presets::optimize_conservatively(&model);
        assert!(result.is_ok());
    }
}
