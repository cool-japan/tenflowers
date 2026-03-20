//! Ultra-Performance Eager Execution Optimization Module
//!
//! This module provides ultra-optimized eager execution to achieve sub-millisecond
//! overhead with advanced SciRS2 ecosystem integration. It focuses on:
//! - Operation caching and intelligent reuse with ML-based optimization
//! - Ultra-efficient memory pool optimization with adaptive strategies
//! - Device synchronization minimization with predictive scheduling
//! - Kernel launch optimization with fusion and batching
//! - Context switching reduction with hardware-aware optimizations
//! - Real-time performance analytics and adaptive tuning
//! - Advanced SIMD acceleration and parallel execution

mod config;
mod engine;
mod memory_pool;
mod reporting;

#[cfg(test)]
mod tests;

pub use config::{
    AdaptiveTuningConfig, CachedOperation, ChunkStrategy, EagerExecutionConfig, ExecutionMetrics,
    GpuAccelerationConfig, GpuSchedulingStrategy, MemoryOptimizationConfig, ModelComplexity,
    OpSignature, ParallelExecutionConfig, PerformanceMonitoringConfig, PoolStrategy,
    PreallocationStrategy, PredictionAlgorithm, SimdExecutionConfig, ThreadPoolStrategy,
    UltraLatencyConfig,
};
pub use engine::EagerExecutionEngine;
pub use reporting::{CacheStatistics, EagerPerformanceReport};

lazy_static::lazy_static! {
    pub static ref EAGER_ENGINE: EagerExecutionEngine =
        EagerExecutionEngine::new(EagerExecutionConfig::default());
}

/// Convenience macro for eager operation execution
#[macro_export]
macro_rules! eager_execute {
    ($op:expr, $inputs:expr, $executor:expr) => {
        $crate::eager_execution::EAGER_ENGINE.execute_operation(
            $op,
            $inputs,
            &std::collections::HashMap::new(),
            $executor,
        )
    };

    ($op:expr, $inputs:expr, $params:expr, $executor:expr) => {
        $crate::eager_execution::EAGER_ENGINE.execute_operation($op, $inputs, $params, $executor)
    };
}
