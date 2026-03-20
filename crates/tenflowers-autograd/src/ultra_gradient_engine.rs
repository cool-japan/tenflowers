//! Ultra-High-Performance Gradient Computation Engine (Legacy Re-export)
//!
//! This module has been refactored into multiple submodules for better organization.
//! The core functionality is now available through the `ultra_gradient` module.

// Re-export all the refactored functionality
pub use crate::ultra_gradient::*;

// Legacy compatibility re-exports
pub use crate::ultra_gradient::{
    CachedGradient, GradientMemoryStats, GradientPerformanceMetrics, GraphOptimizer,
    OptimizationInsights, SimdOpsProcessor, UltraGradientConfig, UltraGradientEngine,
    UltraGradientResult, UltraGradientTapeExt, global_ultra_gradient_engine,
};