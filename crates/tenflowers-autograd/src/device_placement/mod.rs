//! Device Placement Optimization
//!
//! This module provides intelligent device placement for computation graphs,
//! automatically choosing optimal devices (CPU vs GPU) for operations and
//! minimizing data transfers between devices.

mod config;
mod graph_ops;
mod optimizer;
mod types;

// Re-export public types and functions
pub use config::{DevicePlacementConfig, PlacementStrategy};
pub use graph_ops::GraphOperation;
pub use optimizer::DevicePlacementOptimizer;
pub use types::{
    DataTransfer, DeviceCapabilities, OperationProfile, PlacementDecision, PlacementResult,
};
