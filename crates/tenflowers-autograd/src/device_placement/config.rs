//! Device placement configuration types and strategies.

use tenflowers_core::Device;

/// Device placement strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlacementStrategy {
    /// Place all operations on CPU
    CpuOnly,
    /// Place all operations on GPU (if available)
    GpuOnly,
    /// Automatic placement based on operation characteristics
    Auto,
    /// Minimize data transfers between devices
    MinimalTransfer,
    /// Optimize for compute-bound operations on GPU, memory-bound on CPU
    HybridComputeMemory,
}

/// Device placement configuration
#[derive(Debug, Clone)]
pub struct DevicePlacementConfig {
    pub strategy: PlacementStrategy,
    pub available_devices: Vec<Device>,
    pub transfer_cost_weight: f64,
    pub compute_cost_weight: f64,
    pub memory_cost_weight: f64,
    pub enable_cross_device_optimization: bool,
    pub memory_pressure_threshold: f64, // 0.0-1.0, fraction of total memory
    pub enable_pipeline_parallelism: bool,
    pub pipeline_stages: usize,
}

impl Default for DevicePlacementConfig {
    fn default() -> Self {
        Self {
            strategy: PlacementStrategy::Auto,
            available_devices: vec![Device::Cpu],
            transfer_cost_weight: 2.0, // High penalty for transfers
            compute_cost_weight: 1.0,
            memory_cost_weight: 1.5,
            enable_cross_device_optimization: true,
            memory_pressure_threshold: 0.8, // 80% memory usage threshold
            enable_pipeline_parallelism: false,
            pipeline_stages: 1,
        }
    }
}
