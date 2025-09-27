use crate::Layer;
use std::collections::HashMap;
use tenflowers_core::{Device, Result, TensorError};

/// Device placement strategy for layer splitting
#[derive(Debug, Clone, PartialEq)]
pub enum PlacementStrategy {
    /// Automatically choose device based on layer characteristics
    Auto,
    /// Place on specific device
    Device(Device),
    /// Split layer across multiple devices
    Split(Vec<Device>),
    /// Pipeline stage assignment
    Pipeline { stage: usize, device: Device },
}

/// Model parallelism configuration for a model
#[derive(Debug, Clone)]
pub struct ModelParallelConfig {
    /// Device placement for each layer (by layer index)
    pub layer_placement: HashMap<usize, PlacementStrategy>,
    /// Pipeline configuration if using pipeline parallelism
    pub pipeline_config: Option<PipelineConfig>,
    /// Tensor parallelism configuration
    pub tensor_parallel_config: Option<TensorParallelConfig>,
}

/// Pipeline parallelism configuration
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Number of pipeline stages
    pub num_stages: usize,
    /// Micro-batch size for pipeline execution
    pub micro_batch_size: usize,
    /// Number of micro-batches per batch
    pub num_micro_batches: usize,
    /// Devices for each pipeline stage
    pub stage_devices: Vec<Device>,
}

/// Tensor parallelism configuration
#[derive(Debug, Clone)]
pub struct TensorParallelConfig {
    /// Tensor parallel size (number of devices)
    pub tp_size: usize,
    /// Devices for tensor parallelism
    pub devices: Vec<Device>,
    /// Dimension to split along (0 for rows, 1 for columns)
    pub split_dim: usize,
}

/// Layer splitting result
pub struct SplitLayer<T> {
    /// Sub-layers on different devices
    pub sub_layers: Vec<Box<dyn Layer<T>>>,
    /// Device assignment for each sub-layer
    pub device_assignment: Vec<Device>,
    /// Communication pattern for combining results
    pub communication_pattern: CommunicationPattern,
}

/// Communication patterns for combining split layer results
#[derive(Debug, Clone)]
pub enum CommunicationPattern {
    /// Simple concatenation along specified dimension
    Concat { dim: usize },
    /// All-reduce operation (sum results)
    AllReduce,
    /// Gather results to specific device
    Gather { target_device: Device },
    /// Custom communication function
    Custom { pattern_id: String },
}

/// Extended layer trait for model parallelism
pub trait ParallelLayer<T>: Layer<T> {
    /// Get the memory requirements of this layer
    fn memory_requirements(&self) -> Result<MemoryRequirements>;

    /// Split this layer across multiple devices
    fn split_across_devices(&self, devices: &[Device]) -> Result<SplitLayer<T>>;

    /// Get optimal device placement for this layer
    fn suggest_placement(&self, available_devices: &[Device]) -> Result<PlacementStrategy>;

    /// Move layer to specific device
    fn to_device(&mut self, device: &Device) -> Result<()>;

    /// Get current device placement
    fn current_device(&self) -> Option<Device>;

    /// Check if layer can be split
    fn can_split(&self) -> bool;

    /// Get computation intensity (FLOPs per parameter)
    fn computation_intensity(&self) -> f64;
}

/// Memory requirements for a layer
#[derive(Debug, Clone)]
pub struct MemoryRequirements {
    /// Parameter memory in bytes
    pub parameter_memory: usize,
    /// Activation memory in bytes (forward pass)
    pub activation_memory: usize,
    /// Gradient memory in bytes (backward pass)
    pub gradient_memory: usize,
    /// Temporary buffer memory in bytes
    pub temp_memory: usize,
}

impl MemoryRequirements {
    /// Total memory requirement
    pub fn total(&self) -> usize {
        self.parameter_memory + self.activation_memory + self.gradient_memory + self.temp_memory
    }
}

/// Model parallelism coordinator
pub struct ModelParallelCoordinator {
    config: ModelParallelConfig,
    layer_assignments: HashMap<usize, Device>,
    communication_groups: Vec<CommunicationGroup>,
}

/// Communication group for model parallelism
#[derive(Debug, Clone)]
pub struct CommunicationGroup {
    /// Group ID
    pub id: String,
    /// Devices in this group
    pub devices: Vec<Device>,
    /// Group type (pipeline stage, tensor parallel group, etc.)
    pub group_type: GroupType,
}

/// Types of communication groups
#[derive(Debug, Clone)]
pub enum GroupType {
    /// Pipeline stage group
    PipelineStage(usize),
    /// Tensor parallel group
    TensorParallel,
    /// Data parallel group
    DataParallel,
    /// Custom group
    Custom(String),
}

impl ModelParallelCoordinator {
    /// Create new model parallelism coordinator
    pub fn new(config: ModelParallelConfig) -> Self {
        Self {
            config,
            layer_assignments: HashMap::new(),
            communication_groups: Vec::new(),
        }
    }

    /// Assign devices to layers based on configuration
    pub fn assign_devices(&mut self, num_layers: usize) -> Result<()> {
        // Implement device assignment logic
        for layer_idx in 0..num_layers {
            let device = if let Some(strategy) = self.config.layer_placement.get(&layer_idx) {
                match strategy {
                    PlacementStrategy::Device(device) => device.clone(),
                    PlacementStrategy::Pipeline { device, .. } => device.clone(),
                    PlacementStrategy::Auto => {
                        // Auto-assign based on layer index and available devices
                        // For now, use simple round-robin
                        #[cfg(feature = "gpu")]
                        {
                            let device_idx = layer_idx % 4; // Assume 4 devices
                            Device::Gpu(device_idx)
                        }
                        #[cfg(not(feature = "gpu"))]
                        {
                            Device::Cpu
                        }
                    }
                    PlacementStrategy::Split(_) => {
                        // For split layers, assign to first device for now
                        // Full implementation would handle split coordination
                        #[cfg(feature = "gpu")]
                        {
                            Device::Gpu(0)
                        }
                        #[cfg(not(feature = "gpu"))]
                        {
                            Device::Cpu
                        }
                    }
                }
            } else {
                // Default to CPU if no assignment
                Device::Cpu
            };

            self.layer_assignments.insert(layer_idx, device);
        }

        Ok(())
    }

    /// Get device assignment for a specific layer
    pub fn get_layer_device(&self, layer_idx: usize) -> Option<&Device> {
        self.layer_assignments.get(&layer_idx)
    }

    /// Setup communication groups for the parallel execution
    pub fn setup_communication_groups(&mut self) -> Result<()> {
        // Setup pipeline stage groups if using pipeline parallelism
        if let Some(pipeline_config) = &self.config.pipeline_config {
            for stage in 0..pipeline_config.num_stages {
                let group = CommunicationGroup {
                    id: format!("pipeline_stage_{stage}"),
                    devices: vec![pipeline_config.stage_devices[stage].clone()],
                    group_type: GroupType::PipelineStage(stage),
                };
                self.communication_groups.push(group);
            }
        }

        // Setup tensor parallel groups if using tensor parallelism
        if let Some(tp_config) = &self.config.tensor_parallel_config {
            let group = CommunicationGroup {
                id: "tensor_parallel".to_string(),
                devices: tp_config.devices.clone(),
                group_type: GroupType::TensorParallel,
            };
            self.communication_groups.push(group);
        }

        Ok(())
    }
}

/// Default implementation for ModelParallelConfig
impl Default for ModelParallelConfig {
    fn default() -> Self {
        Self {
            layer_placement: HashMap::new(),
            pipeline_config: None,
            tensor_parallel_config: None,
        }
    }
}

/// Utility functions for model parallelism
pub mod utils {
    use super::*;

    /// Calculate optimal layer placement based on memory constraints
    pub fn calculate_optimal_placement<T>(
        layers: &[&dyn ParallelLayer<T>],
        devices: &[Device],
        memory_constraints: &HashMap<Device, usize>,
    ) -> Result<HashMap<usize, PlacementStrategy>> {
        let mut placement = HashMap::new();
        let mut device_memory_used: HashMap<Device, usize> = HashMap::new();

        // Initialize device memory usage
        for device in devices {
            device_memory_used.insert(device.clone(), 0);
        }

        // Assign each layer to best fitting device
        for (layer_idx, layer) in layers.iter().enumerate() {
            let requirements = layer.memory_requirements()?;
            let suggested = layer.suggest_placement(devices)?;

            // Find device with sufficient memory
            let assigned_device = match suggested {
                PlacementStrategy::Device(device) => {
                    if let Some(&constraint) = memory_constraints.get(&device) {
                        let used = device_memory_used.get(&device).unwrap_or(&0);
                        if used + requirements.total() <= constraint {
                            Some(device)
                        } else {
                            None
                        }
                    } else {
                        Some(device)
                    }
                }
                PlacementStrategy::Auto => {
                    // Find device with most available memory
                    devices
                        .iter()
                        .filter_map(|device| {
                            let constraint = memory_constraints.get(device)?;
                            let used = device_memory_used.get(device).unwrap_or(&0);
                            if used + requirements.total() <= *constraint {
                                Some((device, constraint - used))
                            } else {
                                None
                            }
                        })
                        .max_by_key(|(_, available)| *available)
                        .map(|(device, _)| device.clone())
                }
                _ => devices.first().cloned(), // Fallback
            };

            if let Some(device) = assigned_device {
                *device_memory_used.get_mut(&device).unwrap() += requirements.total();
                placement.insert(layer_idx, PlacementStrategy::Device(device));
            } else {
                return Err(TensorError::allocation_error_simple(
                    "Cannot find device with sufficient memory for layer".to_string(),
                ));
            }
        }

        Ok(placement)
    }

    /// Create pipeline configuration from layer assignments
    pub fn create_pipeline_config(
        layer_devices: &HashMap<usize, Device>,
        micro_batch_size: usize,
    ) -> Result<PipelineConfig> {
        // Group consecutive layers on same device into stages
        let mut stages = Vec::new();
        let mut current_stage_device = None;
        let mut stage_devices = Vec::new();

        let mut sorted_layers: Vec<_> = layer_devices.iter().collect();
        sorted_layers.sort_by_key(|(idx, _)| *idx);

        for (_, device) in sorted_layers {
            if current_stage_device.as_ref() != Some(device) {
                if let Some(stage_device) = current_stage_device {
                    stages.push(stage_device);
                }
                current_stage_device = Some(device.clone());
                stage_devices.push(device.clone());
            }
        }

        if let Some(device) = current_stage_device {
            stages.push(device);
        }

        Ok(PipelineConfig {
            num_stages: stage_devices.len(),
            micro_batch_size,
            num_micro_batches: 4, // Default to 4 micro-batches
            stage_devices,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_parallel_config_creation() {
        let config = ModelParallelConfig::default();
        assert!(config.layer_placement.is_empty());
        assert!(config.pipeline_config.is_none());
        assert!(config.tensor_parallel_config.is_none());
    }

    #[test]
    fn test_pipeline_config() {
        let devices = vec![
            #[cfg(feature = "gpu")]
            Device::Gpu(0),
            #[cfg(not(feature = "gpu"))]
            Device::Cpu,
            #[cfg(feature = "gpu")]
            Device::Gpu(1),
            #[cfg(not(feature = "gpu"))]
            Device::Cpu,
            #[cfg(feature = "gpu")]
            Device::Gpu(2),
            #[cfg(not(feature = "gpu"))]
            Device::Cpu,
        ];
        let config = PipelineConfig {
            num_stages: 3,
            micro_batch_size: 8,
            num_micro_batches: 4,
            stage_devices: devices,
        };

        assert_eq!(config.num_stages, 3);
        assert_eq!(config.micro_batch_size, 8);
        assert_eq!(config.num_micro_batches, 4);
    }

    #[test]
    fn test_memory_requirements() {
        let req = MemoryRequirements {
            parameter_memory: 1000,
            activation_memory: 2000,
            gradient_memory: 1000,
            temp_memory: 500,
        };

        assert_eq!(req.total(), 4500);
    }
}
