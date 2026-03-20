//! Kernel fusion type definitions.
//!
//! Split from the original `types.rs` into focused sub-modules for policy
//! compliance (< 2000 lines per file).

pub mod config;
pub mod fusion_types;
pub mod kernel_types;
pub mod metrics;
pub mod patterns;

// Re-export all public items from sub-modules so callers see no change.
pub use config::{
    FusionConstraints, HardwareConfig, MemoryAccessPattern, MemoryLayout, MultiPrecisionMode,
    Precision, SimdConfig,
};
pub use fusion_types::FusedOperation;
pub use kernel_types::{FusableOp, ParallelizationStrategy, SimdInstructionSet};
pub use metrics::{
    AdaptiveFusionStrategy, AdaptiveThresholds, OptimizationLevel, PerformanceDataPoint,
    PerformanceMetrics, PerformanceProfile,
};
pub use patterns::{
    ComputeIntensity, FusedOperationPattern, GpuVendorHints, KernelFusionManager,
    UltraSophisticatedFusionScheduler,
};
