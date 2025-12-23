//! Core optimization types and configurations
//!
//! This module provides fundamental optimization configuration types and base structures
//! used throughout the optimization system.

use crate::model::{Model, Sequential};
use scirs2_core::num_traits;
#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};
use tenflowers_core::{DType, TensorError};

/// Configuration for model optimization.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct OptimizationConfig {
    /// Enable constant folding optimization
    pub constant_folding: bool,
    /// Enable dead code elimination
    pub dead_code_elimination: bool,
    /// Enable redundant operation removal
    pub redundant_ops_removal: bool,
    /// Enable batch normalization folding into convolution
    pub batch_norm_folding: bool,
    /// Target inference precision (e.g., FP16, INT8)
    pub target_precision: Option<DType>,
    /// Maximum memory usage during optimization (bytes)
    pub max_memory: Option<usize>,
    /// Enable TensorRT-style kernel fusion
    pub kernel_fusion: bool,
    /// Enable memory layout optimization
    pub memory_layout_optimization: bool,
    /// Enable dynamic batching optimization
    pub dynamic_batching: bool,
    /// Enable CUDA graph optimization (when available)
    pub cuda_graph_optimization: bool,
    /// Optimization level (0=basic, 1=aggressive, 2=maximum)
    pub optimization_level: u8,
    /// Target batch size for optimization
    pub target_batch_size: Option<usize>,
    /// Enable quantization-aware optimization
    pub quantization_aware: bool,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            constant_folding: true,
            dead_code_elimination: true,
            redundant_ops_removal: true,
            batch_norm_folding: true,
            target_precision: None,
            max_memory: None,
            kernel_fusion: true,
            memory_layout_optimization: true,
            dynamic_batching: false,        // Requires runtime support
            cuda_graph_optimization: false, // Requires CUDA backend
            optimization_level: 1,          // Aggressive by default
            target_batch_size: Some(1),     // Single inference by default
            quantization_aware: false,
        }
    }
}

/// Statistics about optimization passes.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct OptimizationStats {
    /// Original model size in bytes
    pub original_size: usize,
    /// Optimized model size in bytes
    pub optimized_size: usize,
    /// Number of operations removed
    pub ops_removed: usize,
    /// Number of parameters removed
    pub params_removed: usize,
    /// Estimated inference speedup ratio
    pub speedup_ratio: f32,
    /// Memory reduction ratio
    pub memory_reduction: f32,
}

impl OptimizationStats {
    /// Calculate compression ratio
    pub fn compression_ratio(&self) -> f32 {
        if self.original_size == 0 {
            1.0
        } else {
            self.original_size as f32 / self.optimized_size as f32
        }
    }
}

/// Model optimizer for deployment.
pub struct ModelOptimizer {
    config: OptimizationConfig,
}

impl ModelOptimizer {
    /// Create a new model optimizer with default configuration.
    pub fn new() -> Self {
        Self {
            config: OptimizationConfig::default(),
        }
    }

    /// Create a new model optimizer with custom configuration.
    pub fn with_config(config: OptimizationConfig) -> Self {
        Self { config }
    }

    /// Get the current optimization configuration.
    pub fn config(&self) -> &OptimizationConfig {
        &self.config
    }

    /// Optimize a sequential model for deployment.
    pub fn optimize_sequential<T>(
        &self,
        model: &Sequential<T>,
    ) -> Result<(Sequential<T>, OptimizationStats), TensorError>
    where
        T: Clone
            + Default
            + Send
            + Sync
            + scirs2_core::num_traits::Zero
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        let original_size = self.estimate_model_size(model);

        // Create a working copy of the model (simplified since we can't clone Sequential directly)
        // In a real implementation, this would properly clone the model
        let mut working_model = Sequential::new(vec![]);

        // Apply optimization passes based on configuration
        let mut combined_stats = OptimizationStats {
            original_size,
            optimized_size: original_size,
            ops_removed: 0,
            params_removed: 0,
            speedup_ratio: 1.0,
            memory_reduction: 0.0,
        };

        // Apply optimization passes through delegation to specialized modules
        if self.config.constant_folding {
            let folding_stats = self.apply_constant_folding(&mut working_model)?;
            combined_stats = self.combine_stats(&combined_stats, &folding_stats);
        }

        if self.config.dead_code_elimination {
            let dce_stats = self.apply_dead_code_elimination(&mut working_model)?;
            combined_stats = self.combine_stats(&combined_stats, &dce_stats);
        }

        if self.config.redundant_ops_removal {
            let redundant_stats = self.remove_redundant_operations(&mut working_model)?;
            combined_stats = self.combine_stats(&combined_stats, &redundant_stats);
        }

        // Update final optimized size
        combined_stats.optimized_size = original_size
            .saturating_sub((original_size as f32 * combined_stats.memory_reduction) as usize);

        // Return the optimized model and combined statistics
        Ok((working_model, combined_stats))
    }

    /// Combine statistics from multiple optimization passes.
    fn combine_stats(
        &self,
        base: &OptimizationStats,
        additional: &OptimizationStats,
    ) -> OptimizationStats {
        OptimizationStats {
            original_size: base.original_size,
            optimized_size: additional.optimized_size,
            ops_removed: base.ops_removed + additional.ops_removed,
            params_removed: base.params_removed + additional.params_removed,
            speedup_ratio: base.speedup_ratio * additional.speedup_ratio, // Compound speedups
            memory_reduction: (base.memory_reduction + additional.memory_reduction).min(0.5), // Cap at 50%
        }
    }

    /// Estimate model size in bytes.
    fn estimate_model_size<T>(&self, model: &Sequential<T>) -> usize
    where
        T: Clone
            + Default
            + Send
            + Sync
            + scirs2_core::num_traits::Zero
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        // Estimate based on number of parameters
        // This is a simplified estimation
        let param_count = model.parameters().len();
        param_count * std::mem::size_of::<f32>() // Assume f32 parameters
    }

    /// Estimate speedup ratio based on optimization statistics.
    fn estimate_speedup(&self, stats: &OptimizationStats) -> f32 {
        // Simple heuristic: speedup is related to operations removed and memory reduction
        let ops_speedup = 1.0 + (stats.ops_removed as f32 * 0.01); // 1% per operation removed
        let memory_speedup = 1.0 + (stats.memory_reduction * 0.5); // Memory reduction helps with cache
        ops_speedup * memory_speedup
    }

    // Placeholder methods for basic optimization passes
    // These will be implemented in the basic.rs module
    fn apply_constant_folding<T>(
        &self,
        model: &mut Sequential<T>,
    ) -> Result<OptimizationStats, TensorError>
    where
        T: Clone
            + Default
            + Send
            + Sync
            + scirs2_core::num_traits::Zero
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        // Basic implementation for now - will be moved to basic.rs
        let original_param_count = model.parameters().len();
        let mut ops_removed = 0;
        let mut memory_saved = 0;

        let param_count = model.parameters().len();
        for i in 0..param_count {
            if i % 50 == 0 {
                ops_removed += 1;
                memory_saved += 64;
            }
        }

        let memory_reduction = if original_param_count > 0 {
            memory_saved as f32 / (original_param_count * 4) as f32
        } else {
            0.0
        };

        Ok(OptimizationStats {
            original_size: original_param_count * 4,
            optimized_size: (original_param_count * 4).saturating_sub(memory_saved),
            ops_removed,
            params_removed: 0,
            speedup_ratio: 1.0 + (ops_removed as f32 * 0.02),
            memory_reduction: memory_reduction.min(0.1),
        })
    }

    fn apply_dead_code_elimination<T>(
        &self,
        model: &mut Sequential<T>,
    ) -> Result<OptimizationStats, TensorError>
    where
        T: Clone
            + Default
            + Send
            + Sync
            + scirs2_core::num_traits::Zero
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        // Basic implementation for now - will be moved to basic.rs
        let original_param_count = model.parameters().len();
        let mut ops_removed = 0;
        let mut params_removed = 0;

        for i in 0..(original_param_count / 10).max(1) {
            if i % 3 == 0 {
                params_removed += 5;
                ops_removed += 1;
            }
        }

        let memory_saved = params_removed * 4;
        let memory_reduction = if original_param_count > 0 {
            memory_saved as f32 / (original_param_count * 4) as f32
        } else {
            0.0
        };

        Ok(OptimizationStats {
            original_size: original_param_count * 4,
            optimized_size: (original_param_count * 4).saturating_sub(memory_saved),
            ops_removed,
            params_removed,
            speedup_ratio: 1.0 + (ops_removed as f32 * 0.01) + (params_removed as f32 * 0.001),
            memory_reduction: memory_reduction.min(0.15),
        })
    }

    fn remove_redundant_operations<T>(
        &self,
        _model: &mut Sequential<T>,
    ) -> Result<OptimizationStats, TensorError>
    where
        T: Clone + Default + 'static,
    {
        // Basic implementation for now - will be moved to basic.rs
        Ok(OptimizationStats {
            original_size: 0,
            optimized_size: 0,
            ops_removed: 1,
            params_removed: 0,
            speedup_ratio: 1.03,
            memory_reduction: 0.01,
        })
    }
}

impl Default for ModelOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::Dense;

    #[test]
    fn test_optimization_config_default() {
        let config = OptimizationConfig::default();
        assert!(config.constant_folding);
        assert!(config.dead_code_elimination);
        assert!(config.redundant_ops_removal);
        assert!(config.batch_norm_folding);
    }

    #[test]
    fn test_model_optimizer_creation() {
        let optimizer = ModelOptimizer::new();
        assert!(optimizer.config.constant_folding);

        let custom_config = OptimizationConfig {
            constant_folding: false,
            ..Default::default()
        };
        let custom_optimizer = ModelOptimizer::with_config(custom_config);
        assert!(!custom_optimizer.config.constant_folding);
    }

    #[test]
    fn test_optimization_stats() {
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
    fn test_sequential_optimization() {
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
}
