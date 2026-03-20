//! Device placement: strategies, manager, and graph optimizer.
//!
//! Submodules:
//! - `types`: Enums and structs (PlacementStrategy, OpInfo, PrecisionType, etc.)
//! - `manager`: DevicePlacement manager and global convenience functions
//! - `optimizer`: GraphPlacementOptimizer

pub mod manager;
pub mod optimizer;
pub mod types;

mod tests;

pub use manager::{
    choose_device_for_op, estimate_flops, estimate_memory_usage, get_placement_manager,
    set_placement_strategy, DevicePlacement,
};
pub use optimizer::GraphPlacementOptimizer;
pub use types::{
    CostWeights, DeviceCapabilities, GraphOpInfo, OpCategory, OpInfo, OptimizationStats,
    PlacementCost, PlacementStrategy, PrecisionType,
};
