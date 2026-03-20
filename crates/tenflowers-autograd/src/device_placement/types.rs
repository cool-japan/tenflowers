//! Data types for device placement optimization.

use std::collections::{HashMap, HashSet};
use tenflowers_core::Device;

/// Operation characteristics for device placement
#[derive(Debug, Clone)]
pub struct OperationProfile {
    pub operation_name: String,
    pub compute_intensity: f64,       // FLOPs per byte
    pub memory_bandwidth_usage: f64,  // Bytes per second
    pub gpu_acceleration_factor: f64, // Speedup on GPU vs CPU
    pub supports_gpu: bool,
    pub preferred_device: Option<Device>,
    pub tensor_sizes: Vec<usize>, // Input/output tensor sizes
}

/// Device capabilities and performance characteristics
#[derive(Debug, Clone)]
pub struct DeviceCapabilities {
    pub compute_power_gflops: f64,
    pub memory_bandwidth_gb_s: f64,
    pub memory_capacity_gb: f64,
    pub transfer_bandwidth_gb_s: f64, // Bandwidth to other devices
    pub supports_operations: HashSet<String>,
    pub current_memory_usage_gb: f64, // Current memory pressure
    pub peak_memory_usage_gb: f64,    // Peak memory usage observed
}

/// Placement decision for a single operation
#[derive(Debug, Clone)]
pub struct PlacementDecision {
    pub operation_id: String,
    pub chosen_device: Device,
    pub estimated_cost: f64,
    pub transfer_requirements: Vec<DataTransfer>,
    pub reasoning: String,
}

/// Data transfer requirement between devices
#[derive(Debug, Clone)]
pub struct DataTransfer {
    pub from_device: Device,
    pub to_device: Device,
    pub data_size_bytes: usize,
    pub estimated_time_ms: f64,
}

/// Device placement result
#[derive(Debug, Clone)]
pub struct PlacementResult {
    pub decisions: Vec<PlacementDecision>,
    pub total_estimated_cost: f64,
    pub total_transfer_time_ms: f64,
    pub device_utilization: HashMap<Device, f64>,
}
