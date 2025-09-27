//! Device placement optimization pass
//!
//! This module provides device placement optimization for multi-GPU efficiency
//! and performance, including various placement strategies.

use super::passes::{get_node_inputs, OptimizationPass};
use crate::graph::{Graph, NodeId};
use crate::Result;
use std::collections::HashMap;

/// Device placement optimization pass
/// Optimizes device placement for multi-GPU efficiency and performance
pub struct DevicePlacementOptimizationPass {
    available_devices: Vec<crate::device::Device>,
    placement_strategy: PlacementStrategy,
}

/// Strategies for device placement optimization
#[derive(Debug, Clone)]
pub enum PlacementStrategy {
    /// Minimize communication between devices
    MinimizeCommunication,
    /// Balance computational load across devices
    LoadBalancing,
    /// Optimize for memory usage
    MemoryOptimized,
    /// Hybrid strategy considering all factors
    Hybrid,
}

/// Information about an operation's characteristics for placement decisions
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct OperationProfile {
    pub compute_intensity: f32,  // FLOPs per byte of data
    pub memory_usage: usize,     // Estimated memory usage in bytes
    pub parallelizable: bool,    // Can be parallelized across devices
    pub gpu_optimized: bool,     // Performs better on GPU
    pub communication_cost: f32, // Cost of data transfers
}

impl Default for DevicePlacementOptimizationPass {
    fn default() -> Self {
        Self::new()
    }
}

impl DevicePlacementOptimizationPass {
    pub fn new() -> Self {
        Self {
            available_devices: Self::detect_available_devices(),
            placement_strategy: PlacementStrategy::Hybrid,
        }
    }

    pub fn with_strategy(mut self, strategy: PlacementStrategy) -> Self {
        self.placement_strategy = strategy;
        self
    }

    pub fn with_devices(mut self, devices: Vec<crate::device::Device>) -> Self {
        self.available_devices = devices;
        self
    }

    fn detect_available_devices() -> Vec<crate::device::Device> {
        #[cfg(feature = "gpu")]
        let mut devices = vec![crate::device::Device::Cpu];
        #[cfg(not(feature = "gpu"))]
        let devices = vec![crate::device::Device::Cpu];

        #[cfg(feature = "gpu")]
        {
            // In a real implementation, query available GPUs
            // For now, assume at least one GPU is available
            devices.push(crate::device::Device::Gpu(0));
            devices.push(crate::device::Device::Gpu(1));
        }

        devices
    }

    /// Analyzes operation characteristics for placement decisions
    fn analyze_operation(&self, node: &crate::graph::GraphNode) -> OperationProfile {
        match &node.op_type {
            crate::graph::NodeType::Operation(op_name) => {
                match op_name.as_str() {
                    // High compute intensity operations - prefer GPU
                    "MatMul" | "Conv2D" | "Conv3D" => OperationProfile {
                        compute_intensity: 10.0,
                        memory_usage: 1024 * 1024, // 1MB estimate
                        parallelizable: true,
                        gpu_optimized: true,
                        communication_cost: 2.0,
                    },
                    // Medium compute operations
                    "Add" | "Mul" | "Sub" | "Div" => OperationProfile {
                        compute_intensity: 1.0,
                        memory_usage: 64 * 1024, // 64KB estimate
                        parallelizable: true,
                        gpu_optimized: true,
                        communication_cost: 0.5,
                    },
                    // Activation functions - prefer GPU for large tensors
                    "ReLU" | "Sigmoid" | "Tanh" | "Softmax" => OperationProfile {
                        compute_intensity: 0.5,
                        memory_usage: 32 * 1024, // 32KB estimate
                        parallelizable: true,
                        gpu_optimized: true,
                        communication_cost: 0.3,
                    },
                    // Reduction operations - communication intensive
                    "Sum" | "Mean" | "Max" | "Min" => OperationProfile {
                        compute_intensity: 0.8,
                        memory_usage: 16 * 1024, // 16KB estimate
                        parallelizable: false,
                        gpu_optimized: true,
                        communication_cost: 5.0,
                    },
                    // Reshape/transpose operations - memory bound
                    "Reshape" | "Transpose" => OperationProfile {
                        compute_intensity: 0.1,
                        memory_usage: 128 * 1024, // 128KB estimate
                        parallelizable: false,
                        gpu_optimized: false,
                        communication_cost: 1.0,
                    },
                    // Control flow - prefer CPU
                    "If" | "While" | "Switch" => OperationProfile {
                        compute_intensity: 0.1,
                        memory_usage: 1024, // 1KB estimate
                        parallelizable: false,
                        gpu_optimized: false,
                        communication_cost: 0.1,
                    },
                    // Default profile
                    _ => OperationProfile {
                        compute_intensity: 1.0,
                        memory_usage: 64 * 1024,
                        parallelizable: true,
                        gpu_optimized: false,
                        communication_cost: 1.0,
                    },
                }
            }
            crate::graph::NodeType::Constant => OperationProfile {
                compute_intensity: 0.0,
                memory_usage: 4 * 1024, // 4KB estimate
                parallelizable: false,
                gpu_optimized: false,
                communication_cost: 0.1,
            },
            crate::graph::NodeType::Variable { .. } => OperationProfile {
                compute_intensity: 0.0,
                memory_usage: 64 * 1024, // 64KB estimate
                parallelizable: false,
                gpu_optimized: false,
                communication_cost: 1.0,
            },
            crate::graph::NodeType::Placeholder { .. } => OperationProfile {
                compute_intensity: 0.0,
                memory_usage: 32 * 1024, // 32KB estimate
                parallelizable: false,
                gpu_optimized: false,
                communication_cost: 0.5,
            },
        }
    }

    /// Computes the optimal device placement for a node
    fn compute_optimal_placement(
        &self,
        graph: &Graph,
        node_id: NodeId,
        current_placements: &HashMap<NodeId, crate::device::Device>,
    ) -> crate::device::Device {
        let node = graph.get_node(node_id).unwrap();
        let profile = self.analyze_operation(node);

        match self.placement_strategy {
            PlacementStrategy::MinimizeCommunication => {
                self.minimize_communication_placement(graph, node_id, current_placements, &profile)
            }
            PlacementStrategy::LoadBalancing => {
                self.load_balancing_placement(current_placements, &profile)
            }
            PlacementStrategy::MemoryOptimized => self.memory_optimized_placement(&profile),
            PlacementStrategy::Hybrid => {
                self.hybrid_placement(graph, node_id, current_placements, &profile)
            }
        }
    }

    fn minimize_communication_placement(
        &self,
        graph: &Graph,
        node_id: NodeId,
        current_placements: &HashMap<NodeId, crate::device::Device>,
        _profile: &OperationProfile,
    ) -> crate::device::Device {
        // Find the most common device among inputs
        let inputs = get_node_inputs(graph, node_id);
        let mut device_votes: HashMap<crate::device::Device, usize> = HashMap::new();

        for input_id in inputs {
            if let Some(&device) = current_placements.get(&input_id) {
                *device_votes.entry(device).or_insert(0) += 1;
            }
        }

        // Return the device with the most input connections
        device_votes
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(device, _)| device)
            .unwrap_or(crate::device::Device::Cpu)
    }

    fn load_balancing_placement(
        &self,
        current_placements: &HashMap<NodeId, crate::device::Device>,
        profile: &OperationProfile,
    ) -> crate::device::Device {
        // Count operations per device
        let mut device_loads: HashMap<crate::device::Device, usize> = HashMap::new();

        for device in &self.available_devices {
            device_loads.insert(*device, 0);
        }

        for &device in current_placements.values() {
            *device_loads.entry(device).or_insert(0) += 1;
        }

        // For high compute operations, prefer less loaded GPU
        if profile.gpu_optimized && profile.compute_intensity > 5.0 {
            #[cfg(feature = "gpu")]
            {
                device_loads
                    .iter()
                    .filter(|(device, _)| matches!(device, crate::device::Device::Gpu(_)))
                    .min_by_key(|(_, &load)| load)
                    .map(|(&device, _)| device)
                    .unwrap_or(crate::device::Device::Cpu)
            }
            #[cfg(not(feature = "gpu"))]
            {
                crate::device::Device::Cpu
            }
        } else {
            // For other operations, choose least loaded device
            device_loads
                .into_iter()
                .min_by_key(|(_, load)| *load)
                .map(|(device, _)| device)
                .unwrap_or(crate::device::Device::Cpu)
        }
    }

    fn memory_optimized_placement(&self, profile: &OperationProfile) -> crate::device::Device {
        // Place high memory usage operations on devices with more memory
        if profile.memory_usage > 512 * 1024 {
            // > 512KB
            // Prefer CPU for very large memory operations
            crate::device::Device::Cpu
        } else if profile.gpu_optimized {
            // Use GPU for smaller, compute-intensive operations
            #[cfg(feature = "gpu")]
            return crate::device::Device::Gpu(0);
            #[cfg(not(feature = "gpu"))]
            return crate::device::Device::Cpu;
        } else {
            crate::device::Device::Cpu
        }
    }

    fn hybrid_placement(
        &self,
        graph: &Graph,
        node_id: NodeId,
        current_placements: &HashMap<NodeId, crate::device::Device>,
        profile: &OperationProfile,
    ) -> crate::device::Device {
        // Weighted scoring system
        let mut scores: HashMap<crate::device::Device, f32> = HashMap::new();

        for device in &self.available_devices {
            let mut score = 0.0;

            // Factor 1: Operation suitability
            match device {
                crate::device::Device::Cpu => {
                    if !profile.gpu_optimized || profile.compute_intensity < 1.0 {
                        score += 3.0;
                    }
                    if profile.memory_usage > 1024 * 1024 {
                        // > 1MB
                        score += 2.0;
                    }
                }
                #[cfg(feature = "gpu")]
                crate::device::Device::Gpu(_) => {
                    if profile.gpu_optimized {
                        score += 5.0;
                    }
                    if profile.compute_intensity > 2.0 {
                        score += 3.0;
                    }
                    if profile.parallelizable {
                        score += 2.0;
                    }
                }
                #[cfg(feature = "rocm")]
                crate::device::Device::Rocm(_) => {
                    if profile.gpu_optimized {
                        score += 5.0;
                    }
                    if profile.compute_intensity > 2.0 {
                        score += 3.0;
                    }
                    if profile.parallelizable {
                        score += 2.0;
                    }
                }
            }

            // Factor 2: Communication cost
            let inputs = get_node_inputs(graph, node_id);
            let mut communication_penalty = 0.0;
            for input_id in inputs {
                if let Some(&input_device) = current_placements.get(&input_id) {
                    if input_device != *device {
                        communication_penalty += profile.communication_cost;
                    }
                }
            }
            score -= communication_penalty;

            // Factor 3: Load balancing
            let device_load = current_placements
                .values()
                .filter(|&&d| d == *device)
                .count() as f32;
            score -= device_load * 0.1; // Small penalty for device load

            scores.insert(*device, score);
        }

        // Return device with highest score
        scores
            .into_iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(device, _)| device)
            .unwrap_or(crate::device::Device::Cpu)
    }

    /// Estimates the cost of data transfer between devices
    #[allow(dead_code)]
    fn estimate_transfer_cost(
        &self,
        from_device: crate::device::Device,
        to_device: crate::device::Device,
        #[allow(unused_variables)] data_size: usize,
    ) -> f32 {
        if from_device == to_device {
            return 0.0;
        }

        match (from_device, to_device) {
            #[cfg(feature = "gpu")]
            (crate::device::Device::Cpu, crate::device::Device::Gpu(_)) => {
                // CPU to GPU transfer - moderate cost
                data_size as f32 * 0.001 // 1ms per KB
            }
            #[cfg(feature = "gpu")]
            (crate::device::Device::Gpu(_), crate::device::Device::Cpu) => {
                // GPU to CPU transfer - moderate cost
                data_size as f32 * 0.001 // 1ms per KB
            }
            #[cfg(feature = "gpu")]
            (crate::device::Device::Gpu(a), crate::device::Device::Gpu(b)) if a != b => {
                // GPU to GPU transfer - lower cost with high-speed interconnect
                data_size as f32 * 0.0005 // 0.5ms per KB
            }
            _ => 0.0,
        }
    }

    /// Optimizes device placement for the entire graph
    fn optimize_graph_placement(&self, graph: &mut Graph) -> bool {
        let mut current_placements: HashMap<NodeId, crate::device::Device> = HashMap::new();
        let mut changed = false;

        // Initialize with current device placements
        for node in graph.nodes() {
            current_placements.insert(node.id, node.device);
        }

        // Get topological order first (collect to avoid borrowing issues)
        let topo_order = match graph.compute_topological_order() {
            Ok(order) => order.to_vec(),
            Err(_) => {
                // Fallback: collect all node IDs
                graph.nodes().map(|node| node.id).collect::<Vec<_>>()
            }
        };

        // Iteratively optimize placement using topological order
        for &node_id in &topo_order {
            // Compute optimal placement (immutable borrow)
            let optimal_device =
                self.compute_optimal_placement(graph, node_id, &current_placements);

            // Update node device (mutable borrow)
            if let Some(node) = graph.get_node_mut(node_id) {
                if node.device != optimal_device {
                    node.device = optimal_device;
                    current_placements.insert(node_id, optimal_device);
                    changed = true;
                }
            }
        }

        // Collect node information for metadata addition (avoid borrowing conflicts)
        let node_profiles: Vec<(NodeId, OperationProfile)> = graph
            .nodes()
            .map(|node| (node.id, self.analyze_operation(node)))
            .collect();

        // Add placement metadata to nodes
        for (node_id, profile) in node_profiles {
            if let Some(node_mut) = graph.get_node_mut(node_id) {
                node_mut.attributes.insert(
                    "compute_intensity".to_string(),
                    crate::graph::AttributeValue::Float(profile.compute_intensity as f64),
                );
                node_mut.attributes.insert(
                    "memory_usage".to_string(),
                    crate::graph::AttributeValue::Int(profile.memory_usage as i64),
                );
                node_mut.attributes.insert(
                    "gpu_optimized".to_string(),
                    crate::graph::AttributeValue::Bool(profile.gpu_optimized),
                );
            }
        }

        changed
    }
}

impl OptimizationPass for DevicePlacementOptimizationPass {
    fn apply(&self, graph: &mut Graph) -> Result<bool> {
        let changed = self.optimize_graph_placement(graph);
        Ok(changed)
    }

    fn name(&self) -> &str {
        "DevicePlacementOptimization"
    }

    fn is_applicable(&self, graph: &Graph) -> bool {
        // Only applicable if we have multiple devices and multiple nodes
        self.available_devices.len() > 1 && graph.node_count() > 0
    }

    fn priority(&self) -> u32 {
        90 // Run after memory optimization but before low-level optimizations
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_placement_optimization_pass() {
        let pass = DevicePlacementOptimizationPass::new();
        assert_eq!(pass.name(), "DevicePlacementOptimization");
        assert_eq!(pass.priority(), 90);

        // Test with empty graph
        let graph = Graph::new();
        assert!(!pass.is_applicable(&graph));

        // Test with single device
        let single_device_pass =
            DevicePlacementOptimizationPass::new().with_devices(vec![crate::device::Device::Cpu]);
        assert!(!single_device_pass.is_applicable(&graph));
    }

    #[test]
    fn test_device_placement_strategies() {
        let pass = DevicePlacementOptimizationPass::new()
            .with_strategy(PlacementStrategy::MinimizeCommunication);
        assert_eq!(pass.name(), "DevicePlacementOptimization");

        let pass =
            DevicePlacementOptimizationPass::new().with_strategy(PlacementStrategy::LoadBalancing);
        assert_eq!(pass.name(), "DevicePlacementOptimization");

        let pass = DevicePlacementOptimizationPass::new()
            .with_strategy(PlacementStrategy::MemoryOptimized);
        assert_eq!(pass.name(), "DevicePlacementOptimization");

        let pass = DevicePlacementOptimizationPass::new().with_strategy(PlacementStrategy::Hybrid);
        assert_eq!(pass.name(), "DevicePlacementOptimization");
    }

    #[test]
    fn test_operation_profiling() {
        let pass = DevicePlacementOptimizationPass::new();

        // Test high compute operation profiling
        let matmul_node = crate::graph::GraphNode {
            id: 1,
            name: "test_matmul".to_string(),
            op_type: crate::graph::NodeType::Operation("MatMul".to_string()),
            inputs: vec![],
            outputs: vec![],
            device: crate::device::Device::Cpu,
            attributes: std::collections::HashMap::new(),
        };

        let profile = pass.analyze_operation(&matmul_node);
        assert_eq!(profile.compute_intensity, 10.0);
        assert!(profile.gpu_optimized);
        assert!(profile.parallelizable);
        assert_eq!(profile.communication_cost, 2.0);

        // Test low compute operation profiling
        let reshape_node = crate::graph::GraphNode {
            id: 2,
            name: "test_reshape".to_string(),
            op_type: crate::graph::NodeType::Operation("Reshape".to_string()),
            inputs: vec![],
            outputs: vec![],
            device: crate::device::Device::Cpu,
            attributes: std::collections::HashMap::new(),
        };

        let profile = pass.analyze_operation(&reshape_node);
        assert_eq!(profile.compute_intensity, 0.1);
        assert!(!profile.gpu_optimized);
        assert!(!profile.parallelizable);

        // Test constant node profiling
        let constant_node = crate::graph::GraphNode {
            id: 3,
            name: "test_constant".to_string(),
            op_type: crate::graph::NodeType::Constant,
            inputs: vec![],
            outputs: vec![],
            device: crate::device::Device::Cpu,
            attributes: std::collections::HashMap::new(),
        };

        let profile = pass.analyze_operation(&constant_node);
        assert_eq!(profile.compute_intensity, 0.0);
        assert!(!profile.gpu_optimized);
        assert!(!profile.parallelizable);
    }

    #[test]
    fn test_transfer_cost_estimation() {
        let pass = DevicePlacementOptimizationPass::new();

        // Same device - no cost
        let cost = pass.estimate_transfer_cost(
            crate::device::Device::Cpu,
            crate::device::Device::Cpu,
            1024,
        );
        assert_eq!(cost, 0.0);

        #[cfg(feature = "gpu")]
        {
            // CPU to GPU transfer
            let cost = pass.estimate_transfer_cost(
                crate::device::Device::Cpu,
                crate::device::Device::Gpu(0),
                1024,
            );
            assert_eq!(cost, 1.024); // 1024 * 0.001

            // GPU to CPU transfer
            let cost = pass.estimate_transfer_cost(
                crate::device::Device::Gpu(0),
                crate::device::Device::Cpu,
                1024,
            );
            assert_eq!(cost, 1.024); // 1024 * 0.001

            // GPU to GPU transfer
            let cost = pass.estimate_transfer_cost(
                crate::device::Device::Gpu(0),
                crate::device::Device::Gpu(1),
                1024,
            );
            assert_eq!(cost, 0.512); // 1024 * 0.0005
        }
    }

    #[test]
    fn test_memory_optimized_placement() {
        let pass = DevicePlacementOptimizationPass::new()
            .with_strategy(PlacementStrategy::MemoryOptimized);

        // High memory operation should prefer CPU
        let high_mem_profile = OperationProfile {
            compute_intensity: 1.0,
            memory_usage: 1024 * 1024, // 1MB
            parallelizable: true,
            gpu_optimized: true,
            communication_cost: 1.0,
        };

        let device = pass.memory_optimized_placement(&high_mem_profile);
        assert_eq!(device, crate::device::Device::Cpu);

        // Low memory, GPU-optimized operation should prefer GPU (if available)
        let low_mem_profile = OperationProfile {
            compute_intensity: 5.0,
            memory_usage: 64 * 1024, // 64KB
            parallelizable: true,
            gpu_optimized: true,
            communication_cost: 1.0,
        };

        let device = pass.memory_optimized_placement(&low_mem_profile);
        #[cfg(feature = "gpu")]
        assert_eq!(device, crate::device::Device::Gpu(0));
        #[cfg(not(feature = "gpu"))]
        assert_eq!(device, crate::device::Device::Cpu);
    }

    #[test]
    fn test_device_detection() {
        let devices = DevicePlacementOptimizationPass::detect_available_devices();

        // Should always have CPU
        assert!(devices.contains(&crate::device::Device::Cpu));

        // May have GPUs if feature is enabled
        #[cfg(feature = "gpu")]
        {
            assert!(devices.len() > 1);
            assert!(devices.contains(&crate::device::Device::Gpu(0)));
        }

        #[cfg(not(feature = "gpu"))]
        {
            assert_eq!(devices.len(), 1);
        }
    }

    #[test]
    fn test_placement_strategy_builder() {
        #[cfg(feature = "gpu")]
        {
            let pass = DevicePlacementOptimizationPass::new()
                .with_strategy(PlacementStrategy::LoadBalancing)
                .with_devices(vec![
                    crate::device::Device::Cpu,
                    crate::device::Device::Gpu(0),
                ]);

            assert_eq!(pass.available_devices.len(), 2);
            assert!(matches!(
                pass.placement_strategy,
                PlacementStrategy::LoadBalancing
            ));
        }

        #[cfg(not(feature = "gpu"))]
        {
            let pass = DevicePlacementOptimizationPass::new()
                .with_strategy(PlacementStrategy::LoadBalancing)
                .with_devices(vec![crate::device::Device::Cpu]);

            assert_eq!(pass.available_devices.len(), 1);
            assert!(matches!(
                pass.placement_strategy,
                PlacementStrategy::LoadBalancing
            ));
        }
    }
}
