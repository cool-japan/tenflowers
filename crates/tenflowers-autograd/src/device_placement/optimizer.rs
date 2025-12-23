//! Device placement optimizer implementation.

use super::config::{DevicePlacementConfig, PlacementStrategy};
use super::graph_ops::{device_name, GraphOperation};
use super::types::{
    DataTransfer, DeviceCapabilities, OperationProfile, PlacementDecision, PlacementResult,
};
use crate::Result;
use std::collections::{HashMap, HashSet, VecDeque};
use tenflowers_core::Device;

/// Device placement optimizer
#[derive(Debug)]
pub struct DevicePlacementOptimizer {
    config: DevicePlacementConfig,
    operation_profiles: HashMap<String, OperationProfile>,
    device_capabilities: HashMap<Device, DeviceCapabilities>,
}
impl DevicePlacementOptimizer {
    /// Create a new device placement optimizer
    pub fn new(config: DevicePlacementConfig) -> Self {
        let mut optimizer = Self {
            config,
            operation_profiles: HashMap::new(),
            device_capabilities: HashMap::new(),
        };

        optimizer.initialize_default_profiles();
        optimizer.initialize_device_capabilities();
        optimizer
    }

    /// Initialize default operation profiles
    fn initialize_default_profiles(&mut self) {
        let profiles = vec![
            // Linear algebra operations - GPU preferred
            OperationProfile {
                operation_name: "MatMul".to_string(),
                compute_intensity: 2.0, // High compute
                memory_bandwidth_usage: 1.0,
                gpu_acceleration_factor: 10.0,
                supports_gpu: true,
                #[cfg(feature = "gpu")]
                preferred_device: Some(Device::Gpu(0)),
                #[cfg(not(feature = "gpu"))]
                preferred_device: None,
                tensor_sizes: vec![],
            },
            OperationProfile {
                operation_name: "Conv2D".to_string(),
                compute_intensity: 3.0, // Very high compute
                memory_bandwidth_usage: 1.5,
                gpu_acceleration_factor: 15.0,
                supports_gpu: true,
                #[cfg(feature = "gpu")]
                preferred_device: Some(Device::Gpu(0)),
                #[cfg(not(feature = "gpu"))]
                preferred_device: None,
                tensor_sizes: vec![],
            },
            // Element-wise operations - depends on size
            OperationProfile {
                operation_name: "Add".to_string(),
                compute_intensity: 0.1,      // Low compute
                memory_bandwidth_usage: 2.0, // High memory bandwidth
                gpu_acceleration_factor: 2.0,
                supports_gpu: true,
                preferred_device: None, // Size-dependent
                tensor_sizes: vec![],
            },
            OperationProfile {
                operation_name: "Mul".to_string(),
                compute_intensity: 0.1,
                memory_bandwidth_usage: 2.0,
                gpu_acceleration_factor: 2.0,
                supports_gpu: true,
                preferred_device: None,
                tensor_sizes: vec![],
            },
            // Activation functions - GPU preferred for large tensors
            OperationProfile {
                operation_name: "ReLU".to_string(),
                compute_intensity: 0.05,
                memory_bandwidth_usage: 1.0,
                gpu_acceleration_factor: 3.0,
                supports_gpu: true,
                preferred_device: None,
                tensor_sizes: vec![],
            },
            OperationProfile {
                operation_name: "Sigmoid".to_string(),
                compute_intensity: 0.2,
                memory_bandwidth_usage: 1.0,
                gpu_acceleration_factor: 4.0,
                supports_gpu: true,
                preferred_device: None,
                tensor_sizes: vec![],
            },
            // Reduction operations - efficient on GPU
            OperationProfile {
                operation_name: "Sum".to_string(),
                compute_intensity: 0.1,
                memory_bandwidth_usage: 1.0,
                gpu_acceleration_factor: 5.0,
                supports_gpu: true,
                #[cfg(feature = "gpu")]
                preferred_device: Some(Device::Gpu(0)),
                #[cfg(not(feature = "gpu"))]
                preferred_device: None,
                tensor_sizes: vec![],
            },
            // Memory operations - CPU often better
            OperationProfile {
                operation_name: "Reshape".to_string(),
                compute_intensity: 0.0,
                memory_bandwidth_usage: 1.0,
                gpu_acceleration_factor: 0.8, // CPU slightly better
                supports_gpu: true,
                preferred_device: Some(Device::Cpu),
                tensor_sizes: vec![],
            },
        ];

        for profile in profiles {
            self.operation_profiles
                .insert(profile.operation_name.clone(), profile);
        }
    }

    /// Initialize device capabilities
    fn initialize_device_capabilities(&mut self) {
        // CPU capabilities (conservative estimates)
        let cpu_caps = DeviceCapabilities {
            compute_power_gflops: 100.0,
            memory_bandwidth_gb_s: 50.0,
            memory_capacity_gb: 64.0,
            transfer_bandwidth_gb_s: 20.0,
            supports_operations: [
                "Add",
                "Sub",
                "Mul",
                "Div",
                "MatMul",
                "Conv2D",
                "ReLU",
                "Sigmoid",
                "Tanh",
                "Sum",
                "Mean",
                "Reshape",
                "Transpose",
                "Slice",
            ]
            .iter()
            .map(|s| (*s).to_string())
            .collect(),
            current_memory_usage_gb: 0.0,
            peak_memory_usage_gb: 0.0,
        };
        self.device_capabilities.insert(Device::Cpu, cpu_caps);

        // GPU capabilities (modern GPU estimates)
        #[cfg(feature = "gpu")]
        if self
            .config
            .available_devices
            .iter()
            .any(|d| matches!(d, Device::Gpu(_)))
        {
            let gpu_caps = DeviceCapabilities {
                compute_power_gflops: 10000.0,
                memory_bandwidth_gb_s: 900.0,
                memory_capacity_gb: 24.0,
                transfer_bandwidth_gb_s: 50.0,
                supports_operations: [
                    "Add",
                    "Sub",
                    "Mul",
                    "Div",
                    "MatMul",
                    "Conv2D",
                    "ReLU",
                    "Sigmoid",
                    "Tanh",
                    "Sum",
                    "Mean",
                    "Reshape",
                    "Transpose",
                ]
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
                current_memory_usage_gb: 0.0,
                peak_memory_usage_gb: 0.0,
            };
            self.device_capabilities.insert(Device::Gpu(0), gpu_caps);
        }
    }

    /// Optimize device placement for a computation graph
    pub fn optimize_placement<T>(&self, operations: &[GraphOperation<T>]) -> Result<PlacementResult>
    where
        T: Clone + std::fmt::Debug + 'static,
    {
        match self.config.strategy {
            PlacementStrategy::CpuOnly => self.place_all_cpu(operations),
            PlacementStrategy::GpuOnly => self.place_all_gpu(operations),
            PlacementStrategy::Auto => self.auto_placement(operations),
            PlacementStrategy::MinimalTransfer => self.minimal_transfer_placement(operations),
            PlacementStrategy::HybridComputeMemory => self.hybrid_placement(operations),
        }
    }

    /// Place all operations on CPU
    fn place_all_cpu<T>(&self, operations: &[GraphOperation<T>]) -> Result<PlacementResult>
    where
        T: Clone + std::fmt::Debug + 'static,
    {
        let decisions: Vec<PlacementDecision> = operations
            .iter()
            .map(|op| PlacementDecision {
                operation_id: op.id.clone(),
                chosen_device: Device::Cpu,
                estimated_cost: self.estimate_operation_cost(
                    &op.operation_name,
                    &Device::Cpu,
                    &op.tensor_sizes,
                ),
                transfer_requirements: vec![],
                reasoning: "CPU-only strategy".to_string(),
            })
            .collect();

        let total_cost = decisions.iter().map(|d| d.estimated_cost).sum();
        let mut device_util = HashMap::new();
        device_util.insert(Device::Cpu, 1.0);

        Ok(PlacementResult {
            decisions,
            total_estimated_cost: total_cost,
            total_transfer_time_ms: 0.0,
            device_utilization: device_util,
        })
    }

    /// Place all operations on GPU
    #[cfg(feature = "gpu")]
    fn place_all_gpu<T>(&self, operations: &[GraphOperation<T>]) -> Result<PlacementResult>
    where
        T: Clone + std::fmt::Debug + 'static,
    {
        let gpu_device = Device::Gpu(0);
        let decisions: Vec<PlacementDecision> = operations
            .iter()
            .map(|op| PlacementDecision {
                operation_id: op.id.clone(),
                chosen_device: gpu_device,
                estimated_cost: self.estimate_operation_cost(
                    &op.operation_name,
                    &gpu_device,
                    &op.tensor_sizes,
                ),
                transfer_requirements: vec![],
                reasoning: "GPU-only strategy".to_string(),
            })
            .collect();

        let total_cost = decisions.iter().map(|d| d.estimated_cost).sum();
        let mut device_util = HashMap::new();
        device_util.insert(gpu_device, 1.0);

        Ok(PlacementResult {
            decisions,
            total_estimated_cost: total_cost,
            total_transfer_time_ms: 0.0,
            device_utilization: device_util,
        })
    }

    /// Place all operations on GPU (fallback when GPU not available)
    #[cfg(not(feature = "gpu"))]
    fn place_all_gpu<T>(&self, operations: &[GraphOperation<T>]) -> Result<PlacementResult>
    where
        T: Clone + std::fmt::Debug + 'static,
    {
        // Fallback to CPU when GPU not available
        self.place_all_cpu(operations)
    }

    /// Automatic placement based on operation characteristics
    fn auto_placement<T>(&self, operations: &[GraphOperation<T>]) -> Result<PlacementResult>
    where
        T: Clone + std::fmt::Debug + 'static,
    {
        let mut decisions = Vec::new();
        let mut device_loads: HashMap<Device, f64> = HashMap::new();

        for op in operations {
            let best_device = self.choose_best_device_for_operation(op, &device_loads)?;
            let cost =
                self.estimate_operation_cost(&op.operation_name, &best_device, &op.tensor_sizes);

            // Update device load
            *device_loads.entry(best_device).or_insert(0.0) += cost;

            decisions.push(PlacementDecision {
                operation_id: op.id.clone(),
                chosen_device: best_device,
                estimated_cost: cost,
                transfer_requirements: self.compute_transfer_requirements(
                    op,
                    &best_device,
                    &decisions,
                ),
                reasoning: format!(
                    "Auto-placement: {} chosen for {}",
                    device_name(&best_device),
                    op.operation_name
                ),
            });
        }

        let total_cost = decisions.iter().map(|d| d.estimated_cost).sum();
        let total_transfer_time: f64 = decisions
            .iter()
            .flat_map(|d| &d.transfer_requirements)
            .map(|t| t.estimated_time_ms)
            .sum();

        // Normalize device utilization
        let max_load = device_loads.values().fold(0.0f64, |a, &b| a.max(b));
        let device_utilization = device_loads
            .into_iter()
            .map(|(device, load)| (device, if max_load > 0.0 { load / max_load } else { 0.0 }))
            .collect();

        Ok(PlacementResult {
            decisions,
            total_estimated_cost: total_cost,
            total_transfer_time_ms: total_transfer_time,
            device_utilization,
        })
    }

    /// Placement strategy that minimizes data transfers
    fn minimal_transfer_placement<T>(
        &self,
        operations: &[GraphOperation<T>],
    ) -> Result<PlacementResult>
    where
        T: Clone + std::fmt::Debug + 'static,
    {
        // Use a graph coloring approach to minimize transfers
        let mut decisions = Vec::new();
        let mut operation_to_device: HashMap<String, Device> = HashMap::new();

        // Build dependency graph
        let dependencies = self.build_dependency_graph(operations);

        // Process operations in topological order
        let topo_order = self.topological_sort(operations, &dependencies)?;

        for &op_idx in &topo_order {
            let op = &operations[op_idx];
            let best_device =
                self.choose_device_minimal_transfer(op, &operation_to_device, &dependencies)?;

            operation_to_device.insert(op.id.clone(), best_device);

            let cost =
                self.estimate_operation_cost(&op.operation_name, &best_device, &op.tensor_sizes);
            let transfer_reqs =
                self.compute_transfer_requirements_minimal(op, &best_device, &operation_to_device);

            decisions.push(PlacementDecision {
                operation_id: op.id.clone(),
                chosen_device: best_device,
                estimated_cost: cost,
                transfer_requirements: transfer_reqs,
                reasoning: "Minimal transfer strategy".to_string(),
            });
        }

        let total_cost = decisions.iter().map(|d| d.estimated_cost).sum();
        let total_transfer_time: f64 = decisions
            .iter()
            .flat_map(|d| &d.transfer_requirements)
            .map(|t| t.estimated_time_ms)
            .sum();

        Ok(PlacementResult {
            decisions,
            total_estimated_cost: total_cost,
            total_transfer_time_ms: total_transfer_time,
            device_utilization: HashMap::new(), // Computed separately if needed
        })
    }

    /// Hybrid placement optimizing for compute vs memory bound operations
    fn hybrid_placement<T>(&self, operations: &[GraphOperation<T>]) -> Result<PlacementResult>
    where
        T: Clone + std::fmt::Debug + 'static,
    {
        let mut decisions = Vec::new();

        for op in operations {
            let profile = self
                .operation_profiles
                .get(&op.operation_name)
                .cloned()
                .unwrap_or_else(|| self.default_operation_profile(&op.operation_name));

            let chosen_device = if profile.compute_intensity > 1.0 {
                // Compute-bound -> prefer GPU
                #[cfg(feature = "gpu")]
                {
                    if self
                        .config
                        .available_devices
                        .iter()
                        .any(|d| matches!(d, Device::Gpu(_)))
                    {
                        Device::Gpu(0)
                    } else {
                        Device::Cpu
                    }
                }
                #[cfg(not(feature = "gpu"))]
                {
                    Device::Cpu
                }
            } else {
                // Memory-bound or low compute -> prefer CPU for small tensors, GPU for large
                #[cfg(feature = "gpu")]
                let total_tensor_size: usize = op.tensor_sizes.iter().sum();
                #[cfg(feature = "gpu")]
                {
                    if total_tensor_size > 1024 * 1024 {
                        // > 1MB
                        Device::Gpu(0)
                    } else {
                        Device::Cpu
                    }
                }
                #[cfg(not(feature = "gpu"))]
                {
                    Device::Cpu
                }
            };

            let cost =
                self.estimate_operation_cost(&op.operation_name, &chosen_device, &op.tensor_sizes);

            decisions.push(PlacementDecision {
                operation_id: op.id.clone(),
                chosen_device,
                estimated_cost: cost,
                transfer_requirements: vec![], // Computed separately
                reasoning: format!(
                    "Hybrid strategy: compute_intensity={:.2}",
                    profile.compute_intensity
                ),
            });
        }

        let total_cost = decisions.iter().map(|d| d.estimated_cost).sum();

        Ok(PlacementResult {
            decisions,
            total_estimated_cost: total_cost,
            total_transfer_time_ms: 0.0,
            device_utilization: HashMap::new(),
        })
    }

    /// Choose the best device for a single operation
    fn choose_best_device_for_operation<T>(
        &self,
        op: &GraphOperation<T>,
        device_loads: &HashMap<Device, f64>,
    ) -> Result<Device>
    where
        T: Clone + std::fmt::Debug + 'static,
    {
        let profile = self
            .operation_profiles
            .get(&op.operation_name)
            .cloned()
            .unwrap_or_else(|| self.default_operation_profile(&op.operation_name));

        if let Some(preferred) = profile.preferred_device {
            if self.config.available_devices.contains(&preferred) {
                return Ok(preferred);
            }
        }

        // Evaluate cost on each available device
        let mut best_device = Device::Cpu;
        let mut best_score = f64::INFINITY;

        for device in &self.config.available_devices {
            let compute_cost =
                self.estimate_operation_cost(&op.operation_name, device, &op.tensor_sizes);
            let load_penalty = device_loads.get(device).unwrap_or(&0.0) * 0.1;
            let total_score = compute_cost + load_penalty;

            if total_score < best_score {
                best_score = total_score;
                best_device = *device;
            }
        }

        Ok(best_device)
    }

    /// Estimate the cost of running an operation on a device
    fn estimate_operation_cost(
        &self,
        operation: &str,
        device: &Device,
        tensor_sizes: &[usize],
    ) -> f64 {
        let profile = self
            .operation_profiles
            .get(operation)
            .cloned()
            .unwrap_or_else(|| self.default_operation_profile(operation));

        let device_caps = self
            .device_capabilities
            .get(device)
            .cloned()
            .unwrap_or_else(|| self.default_device_capabilities(device));

        let total_elements: usize = tensor_sizes.iter().sum();
        let total_bytes = total_elements * 4; // Assume f32

        // Compute time = (compute_operations / compute_power) + (memory_accesses / memory_bandwidth)
        let compute_operations = total_elements as f64 * profile.compute_intensity;
        let compute_time = compute_operations / device_caps.compute_power_gflops;

        let memory_time = total_bytes as f64 / (device_caps.memory_bandwidth_gb_s * 1e9);

        let base_time = compute_time + memory_time;

        // Apply GPU acceleration factor
        match device {
            #[cfg(feature = "gpu")]
            Device::Gpu(_) if profile.supports_gpu => base_time / profile.gpu_acceleration_factor,
            _ => base_time,
        }
    }

    /// Create default operation profile for unknown operations
    fn default_operation_profile(&self, operation: &str) -> OperationProfile {
        OperationProfile {
            operation_name: operation.to_string(),
            compute_intensity: 0.5,
            memory_bandwidth_usage: 1.0,
            gpu_acceleration_factor: 2.0,
            supports_gpu: true,
            preferred_device: None,
            tensor_sizes: vec![],
        }
    }

    /// Create default device capabilities
    fn default_device_capabilities(&self, device: &Device) -> DeviceCapabilities {
        #[allow(unreachable_patterns)] // GPU/ROCM patterns unreachable when features are disabled
        match device {
            Device::Cpu => DeviceCapabilities {
                compute_power_gflops: 100.0,
                memory_bandwidth_gb_s: 50.0,
                memory_capacity_gb: 64.0,
                transfer_bandwidth_gb_s: 20.0,
                supports_operations: HashSet::new(),
                current_memory_usage_gb: 0.0,
                peak_memory_usage_gb: 0.0,
            },
            #[cfg(feature = "gpu")]
            Device::Gpu(_) => DeviceCapabilities {
                compute_power_gflops: 10000.0,
                memory_bandwidth_gb_s: 900.0,
                memory_capacity_gb: 24.0,
                transfer_bandwidth_gb_s: 50.0,
                supports_operations: HashSet::new(),
                current_memory_usage_gb: 0.0,
                peak_memory_usage_gb: 0.0,
            },
            #[cfg(feature = "rocm")]
            Device::Rocm(_) => DeviceCapabilities {
                compute_power_gflops: 8000.0,
                memory_bandwidth_gb_s: 800.0,
                memory_capacity_gb: 16.0,
                transfer_bandwidth_gb_s: 45.0,
                supports_operations: HashSet::new(),
                current_memory_usage_gb: 0.0,
                peak_memory_usage_gb: 0.0,
            },
            #[cfg(not(any(feature = "gpu", feature = "rocm")))]
            _ => unreachable!("GPU/ROCM variants should not exist without features"),
        }
    }

    /// Build dependency graph between operations
    fn build_dependency_graph<T>(
        &self,
        operations: &[GraphOperation<T>],
    ) -> HashMap<usize, Vec<usize>>
    where
        T: Clone + std::fmt::Debug + 'static,
    {
        let mut dependencies = HashMap::new();

        for (i, op) in operations.iter().enumerate() {
            let mut deps = Vec::new();
            for (j, other_op) in operations.iter().enumerate() {
                if i != j
                    && op
                        .inputs
                        .iter()
                        .any(|input| other_op.outputs.contains(input))
                {
                    deps.push(j);
                }
            }
            dependencies.insert(i, deps);
        }

        dependencies
    }

    /// Perform topological sort on operations
    fn topological_sort<T>(
        &self,
        operations: &[GraphOperation<T>],
        dependencies: &HashMap<usize, Vec<usize>>,
    ) -> Result<Vec<usize>>
    where
        T: Clone + std::fmt::Debug + 'static,
    {
        let mut in_degree = vec![0; operations.len()];

        // Calculate in-degrees
        for deps in dependencies.values() {
            for &dep in deps {
                in_degree[dep] += 1;
            }
        }

        let mut queue = VecDeque::new();
        for (i, &degree) in in_degree.iter().enumerate() {
            if degree == 0 {
                queue.push_back(i);
            }
        }

        let mut result = Vec::new();

        while let Some(node) = queue.pop_front() {
            result.push(node);

            if let Some(deps) = dependencies.get(&node) {
                for &dep in deps {
                    in_degree[dep] -= 1;
                    if in_degree[dep] == 0 {
                        queue.push_back(dep);
                    }
                }
            }
        }

        if result.len() != operations.len() {
            return Err(tenflowers_core::TensorError::invalid_argument(
                "Circular dependency detected in computation graph".to_string(),
            ));
        }

        Ok(result)
    }

    /// Choose device to minimize transfers
    fn choose_device_minimal_transfer<T>(
        &self,
        op: &GraphOperation<T>,
        existing_placements: &HashMap<String, Device>,
        _dependencies: &HashMap<usize, Vec<usize>>,
    ) -> Result<Device>
    where
        T: Clone + std::fmt::Debug + 'static,
    {
        let mut device_scores: HashMap<Device, f64> = HashMap::new();

        // Initialize with base operation costs
        for device in &self.config.available_devices {
            let base_cost =
                self.estimate_operation_cost(&op.operation_name, device, &op.tensor_sizes);
            device_scores.insert(*device, base_cost);
        }

        // Add transfer penalties
        for input in &op.inputs {
            if let Some(input_device) = existing_placements.get(input) {
                for device in &self.config.available_devices {
                    if device != input_device {
                        // Add transfer cost penalty
                        let transfer_penalty =
                            self.estimate_transfer_cost(input_device, device, 1024); // Estimate
                        *device_scores.get_mut(device).unwrap() +=
                            transfer_penalty * self.config.transfer_cost_weight;
                    }
                }
            }
        }

        // Choose device with minimum score
        let best_device = device_scores
            .into_iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(device, _)| device)
            .unwrap_or(Device::Cpu);

        Ok(best_device)
    }

    /// Estimate cost of transferring data between devices
    fn estimate_transfer_cost(&self, from: &Device, to: &Device, size_bytes: usize) -> f64 {
        if from == to {
            return 0.0;
        }

        let from_caps = self
            .device_capabilities
            .get(from)
            .cloned()
            .unwrap_or_else(|| self.default_device_capabilities(from));

        // Transfer time = size / bandwidth
        size_bytes as f64 / (from_caps.transfer_bandwidth_gb_s * 1e9)
    }

    /// Compute transfer requirements for an operation
    fn compute_transfer_requirements<T>(
        &self,
        _op: &GraphOperation<T>,
        _chosen_device: &Device,
        _previous_decisions: &[PlacementDecision],
    ) -> Vec<DataTransfer>
    where
        T: Clone + std::fmt::Debug + 'static,
    {
        // Simplified implementation - in practice would track tensor locations
        vec![]
    }

    /// Compute transfer requirements for minimal transfer strategy
    fn compute_transfer_requirements_minimal<T>(
        &self,
        _op: &GraphOperation<T>,
        _chosen_device: &Device,
        _existing_placements: &HashMap<String, Device>,
    ) -> Vec<DataTransfer>
    where
        T: Clone + std::fmt::Debug + 'static,
    {
        // Simplified implementation
        vec![]
    }

    /// Update device memory usage statistics
    pub fn update_memory_usage(&mut self, device: &Device, memory_usage_gb: f64) {
        if let Some(capabilities) = self.device_capabilities.get_mut(device) {
            capabilities.current_memory_usage_gb = memory_usage_gb;
            capabilities.peak_memory_usage_gb =
                capabilities.peak_memory_usage_gb.max(memory_usage_gb);
        }
    }

    /// Check if device is under memory pressure
    pub fn is_memory_pressure(&self, device: &Device) -> bool {
        if let Some(capabilities) = self.device_capabilities.get(device) {
            let memory_utilization =
                capabilities.current_memory_usage_gb / capabilities.memory_capacity_gb;
            memory_utilization > self.config.memory_pressure_threshold
        } else {
            false
        }
    }

    /// Advanced cost model considering memory pressure and device characteristics
    pub fn calculate_advanced_cost(
        &self,
        operation_name: &str,
        device: &Device,
        tensor_sizes: &[usize],
        memory_pressure_penalty: f64,
    ) -> f64 {
        let base_cost = self.estimate_operation_cost(operation_name, device, tensor_sizes);

        // Get device capabilities
        let capabilities = self
            .device_capabilities
            .get(device)
            .cloned()
            .unwrap_or_else(|| self.default_device_capabilities(device));

        // Calculate memory pressure factor
        let memory_utilization =
            capabilities.current_memory_usage_gb / capabilities.memory_capacity_gb;
        let memory_factor = if memory_utilization > self.config.memory_pressure_threshold {
            1.0 + memory_pressure_penalty
                * (memory_utilization - self.config.memory_pressure_threshold)
        } else {
            1.0
        };

        // Calculate operation memory requirement
        let operation_memory_gb =
            tensor_sizes.iter().sum::<usize>() as f64 / (1024.0 * 1024.0 * 1024.0);
        let memory_cost = operation_memory_gb * self.config.memory_cost_weight;

        // Combine costs
        base_cost * memory_factor + memory_cost
    }

    /// Placement with memory pressure awareness
    pub fn placement_with_memory_awareness<T>(
        &mut self,
        operations: &[GraphOperation<T>],
    ) -> Result<PlacementResult>
    where
        T: Clone + std::fmt::Debug + 'static,
    {
        let mut decisions = Vec::new();
        let mut device_memory_usage: HashMap<Device, f64> = HashMap::new();

        // Initialize device memory usage
        for device in &self.config.available_devices {
            if let Some(capabilities) = self.device_capabilities.get(device) {
                device_memory_usage.insert(*device, capabilities.current_memory_usage_gb);
            }
        }

        for op in operations {
            let mut best_device = Device::Cpu;
            let mut best_cost = f64::INFINITY;

            // Calculate operation memory requirement
            let operation_memory_gb =
                op.tensor_sizes.iter().sum::<usize>() as f64 / (1024.0 * 1024.0 * 1024.0);

            for device in &self.config.available_devices {
                // Check if operation would fit in device memory
                let current_usage = device_memory_usage.get(device).unwrap_or(&0.0);
                let capabilities = self
                    .device_capabilities
                    .get(device)
                    .cloned()
                    .unwrap_or_else(|| self.default_device_capabilities(device));

                if current_usage + operation_memory_gb > capabilities.memory_capacity_gb {
                    // Skip this device if operation won't fit
                    continue;
                }

                // Calculate cost with memory pressure awareness
                let memory_pressure_penalty = if self.is_memory_pressure(device) {
                    2.0
                } else {
                    0.5
                };
                let cost = self.calculate_advanced_cost(
                    &op.operation_name,
                    device,
                    &op.tensor_sizes,
                    memory_pressure_penalty,
                );

                if cost < best_cost {
                    best_cost = cost;
                    best_device = *device;
                }
            }

            // Update memory usage for chosen device
            let operation_memory_gb =
                op.tensor_sizes.iter().sum::<usize>() as f64 / (1024.0 * 1024.0 * 1024.0);
            *device_memory_usage.entry(best_device).or_insert(0.0) += operation_memory_gb;

            // Update actual device capabilities
            self.update_memory_usage(
                &best_device,
                *device_memory_usage.get(&best_device).unwrap(),
            );

            decisions.push(PlacementDecision {
                operation_id: op.id.clone(),
                chosen_device: best_device,
                estimated_cost: best_cost,
                transfer_requirements: vec![], // Simplified
                reasoning: "Memory-aware placement".to_string(),
            });
        }

        let total_cost = decisions.iter().map(|d| d.estimated_cost).sum();

        Ok(PlacementResult {
            decisions,
            total_estimated_cost: total_cost,
            total_transfer_time_ms: 0.0, // Simplified
            device_utilization: device_memory_usage.into_iter().collect(),
        })
    }

    /// Pipeline parallelism placement
    pub fn pipeline_placement<T>(&self, operations: &[GraphOperation<T>]) -> Result<PlacementResult>
    where
        T: Clone + std::fmt::Debug + 'static,
    {
        if !self.config.enable_pipeline_parallelism {
            return self.auto_placement(operations);
        }

        let mut decisions = Vec::new();
        let operations_per_stage = operations.len().max(1) / self.config.pipeline_stages.max(1);

        for (i, op) in operations.iter().enumerate() {
            let stage = i / operations_per_stage;
            let device_index = stage % self.config.available_devices.len();
            let chosen_device = self.config.available_devices[device_index];

            let cost =
                self.estimate_operation_cost(&op.operation_name, &chosen_device, &op.tensor_sizes);

            decisions.push(PlacementDecision {
                operation_id: op.id.clone(),
                chosen_device,
                estimated_cost: cost,
                transfer_requirements: vec![], // Simplified
                reasoning: format!("Pipeline stage {stage}"),
            });
        }

        let total_cost = decisions.iter().map(|d| d.estimated_cost).sum();

        Ok(PlacementResult {
            decisions,
            total_estimated_cost: total_cost,
            total_transfer_time_ms: 0.0,        // Simplified
            device_utilization: HashMap::new(), // Simplified
        })
    }
}
