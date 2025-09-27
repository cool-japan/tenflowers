//! Subgraph Data Types
//!
//! This module defines the core data structures used to represent subgraphs,
//! operations, tensors, and communication requirements.

pub use crate::device_placement::GraphOperation;
use std::collections::HashMap;

/// A computational subgraph
#[derive(Debug, Clone)]
pub struct Subgraph {
    pub id: String,
    pub operations: Vec<SubgraphOperation>,
    pub inputs: Vec<SubgraphTensor>,
    pub outputs: Vec<SubgraphTensor>,
    pub internal_tensors: Vec<SubgraphTensor>,
    pub estimated_compute_cost: f64,
    pub estimated_memory_usage: usize,
    pub dependencies: Vec<String>, // IDs of subgraphs this depends on
    pub parallelizable_with: Vec<String>, // IDs of subgraphs this can run in parallel with
}

impl Subgraph {
    /// Create a new empty subgraph
    pub fn new(id: String) -> Self {
        Self {
            id,
            operations: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            internal_tensors: Vec::new(),
            estimated_compute_cost: 0.0,
            estimated_memory_usage: 0,
            dependencies: Vec::new(),
            parallelizable_with: Vec::new(),
        }
    }

    /// Add an operation to this subgraph
    pub fn add_operation(&mut self, operation: SubgraphOperation) {
        self.estimated_compute_cost += operation.estimated_flops as f64;
        self.estimated_memory_usage += operation.estimated_memory_bytes;
        self.operations.push(operation);
    }

    /// Add a dependency on another subgraph
    pub fn add_dependency(&mut self, subgraph_id: String) {
        if !self.dependencies.contains(&subgraph_id) {
            self.dependencies.push(subgraph_id);
        }
    }

    /// Mark this subgraph as parallelizable with another
    pub fn add_parallelizable_with(&mut self, subgraph_id: String) {
        if !self.parallelizable_with.contains(&subgraph_id) {
            self.parallelizable_with.push(subgraph_id);
        }
    }

    /// Check if this subgraph depends on another
    pub fn depends_on(&self, subgraph_id: &str) -> bool {
        self.dependencies.contains(&subgraph_id.to_string())
    }

    /// Get the total estimated FLOPs for this subgraph
    pub fn total_flops(&self) -> u64 {
        self.operations.iter().map(|op| op.estimated_flops).sum()
    }

    /// Get the number of operations in this subgraph
    pub fn operation_count(&self) -> usize {
        self.operations.len()
    }
}

/// Operation within a subgraph
#[derive(Debug, Clone)]
pub struct SubgraphOperation {
    pub id: String,
    pub operation_type: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub estimated_flops: u64,
    pub estimated_memory_bytes: usize,
    pub can_be_parallel: bool,
}

impl SubgraphOperation {
    /// Create a new subgraph operation
    pub fn new(
        id: String,
        operation_type: String,
        inputs: Vec<String>,
        outputs: Vec<String>,
    ) -> Self {
        Self {
            id,
            operation_type,
            inputs,
            outputs,
            estimated_flops: 0,
            estimated_memory_bytes: 0,
            can_be_parallel: false,
        }
    }

    /// Set computational complexity estimates
    pub fn with_estimates(mut self, flops: u64, memory_bytes: usize) -> Self {
        self.estimated_flops = flops;
        self.estimated_memory_bytes = memory_bytes;
        self
    }

    /// Set parallel execution capability
    pub fn with_parallel(mut self, can_parallel: bool) -> Self {
        self.can_be_parallel = can_parallel;
        self
    }

    /// Check if this operation is computationally expensive
    pub fn is_expensive(&self) -> bool {
        self.estimated_flops > 1_000_000 || self.estimated_memory_bytes > 1_024_000
    }
}

/// Tensor within or between subgraphs
#[derive(Debug, Clone)]
pub struct SubgraphTensor {
    pub id: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub size_bytes: usize,
    pub producer_subgraph: Option<String>,
    pub consumer_subgraphs: Vec<String>,
}

impl SubgraphTensor {
    /// Create a new subgraph tensor
    pub fn new(id: String, shape: Vec<usize>, dtype: String) -> Self {
        let size_bytes = Self::calculate_size_bytes(&shape, &dtype);
        Self {
            id,
            shape,
            dtype,
            size_bytes,
            producer_subgraph: None,
            consumer_subgraphs: Vec::new(),
        }
    }

    /// Calculate size in bytes based on shape and data type
    fn calculate_size_bytes(shape: &[usize], dtype: &str) -> usize {
        let element_count = shape.iter().product::<usize>();
        let bytes_per_element = match dtype {
            "f32" | "i32" | "u32" => 4,
            "f64" | "i64" | "u64" => 8,
            "f16" | "i16" | "u16" => 2,
            "i8" | "u8" => 1,
            _ => 4, // Default to 4 bytes
        };
        element_count * bytes_per_element
    }

    /// Set the producer subgraph
    pub fn set_producer(&mut self, subgraph_id: String) {
        self.producer_subgraph = Some(subgraph_id);
    }

    /// Add a consumer subgraph
    pub fn add_consumer(&mut self, subgraph_id: String) {
        if !self.consumer_subgraphs.contains(&subgraph_id) {
            self.consumer_subgraphs.push(subgraph_id);
        }
    }

    /// Check if this tensor crosses subgraph boundaries
    pub fn is_cross_subgraph(&self) -> bool {
        self.consumer_subgraphs.len() > 1
            || (self.producer_subgraph.is_some() && !self.consumer_subgraphs.is_empty())
    }

    /// Get the number of elements in this tensor
    pub fn element_count(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Communication requirement between subgraphs
#[derive(Debug, Clone)]
pub struct SubgraphCommunication {
    pub from_subgraph: String,
    pub to_subgraph: String,
    pub tensor: SubgraphTensor,
    pub estimated_transfer_time_ms: f64,
    pub is_critical_path: bool,
}

impl SubgraphCommunication {
    /// Create a new communication requirement
    pub fn new(from_subgraph: String, to_subgraph: String, tensor: SubgraphTensor) -> Self {
        let estimated_transfer_time_ms = Self::estimate_transfer_time(&tensor);
        Self {
            from_subgraph,
            to_subgraph,
            tensor,
            estimated_transfer_time_ms,
            is_critical_path: false,
        }
    }

    /// Estimate transfer time based on tensor size
    fn estimate_transfer_time(tensor: &SubgraphTensor) -> f64 {
        // Simplified model: assume 10 GB/s transfer rate
        let transfer_rate_bytes_per_ms = 10_000_000.0;
        tensor.size_bytes as f64 / transfer_rate_bytes_per_ms
    }

    /// Mark this communication as being on the critical path
    pub fn set_critical_path(&mut self, is_critical: bool) {
        self.is_critical_path = is_critical;
    }

    /// Get the communication cost (transfer time weighted by criticality)
    pub fn communication_cost(&self) -> f64 {
        if self.is_critical_path {
            self.estimated_transfer_time_ms * 2.0
        } else {
            self.estimated_transfer_time_ms
        }
    }
}

/// Result of subgraph extraction
#[derive(Debug, Clone)]
pub struct SubgraphExtractionResult {
    pub subgraphs: Vec<Subgraph>,
    pub communications: Vec<SubgraphCommunication>,
    pub execution_order: Vec<Vec<String>>, // Parallel stages
    pub total_estimated_time_ms: f64,
    pub parallel_efficiency: f64,
    pub memory_efficiency: f64,
    pub total_communication_cost: f64,
    pub load_balance_score: f64,
    pub metadata: HashMap<String, String>,
}

impl SubgraphExtractionResult {
    /// Create a new extraction result
    pub fn new() -> Self {
        Self {
            subgraphs: Vec::new(),
            communications: Vec::new(),
            execution_order: Vec::new(),
            total_estimated_time_ms: 0.0,
            parallel_efficiency: 0.0,
            memory_efficiency: 0.0,
            total_communication_cost: 0.0,
            load_balance_score: 0.0,
            metadata: HashMap::new(),
        }
    }

    /// Add a subgraph to the result
    pub fn add_subgraph(&mut self, subgraph: Subgraph) {
        self.subgraphs.push(subgraph);
    }

    /// Add a communication requirement
    pub fn add_communication(&mut self, communication: SubgraphCommunication) {
        self.total_communication_cost += communication.communication_cost();
        self.communications.push(communication);
    }

    /// Calculate load balance score (1.0 = perfect balance, 0.0 = completely unbalanced)
    pub fn calculate_load_balance(&mut self) {
        if self.subgraphs.is_empty() {
            self.load_balance_score = 0.0;
            return;
        }

        let compute_costs: Vec<f64> = self
            .subgraphs
            .iter()
            .map(|sg| sg.estimated_compute_cost)
            .collect();

        let max_cost = compute_costs.iter().fold(0.0f64, |a, &b| a.max(b));
        let min_cost = compute_costs.iter().fold(f64::INFINITY, |a, &b| a.min(b));

        if max_cost > 0.0 {
            self.load_balance_score = min_cost / max_cost;
        } else {
            self.load_balance_score = 1.0;
        }
    }

    /// Get the number of subgraphs
    pub fn subgraph_count(&self) -> usize {
        self.subgraphs.len()
    }

    /// Get total computational cost across all subgraphs
    pub fn total_compute_cost(&self) -> f64 {
        self.subgraphs
            .iter()
            .map(|sg| sg.estimated_compute_cost)
            .sum()
    }
}

impl Default for SubgraphExtractionResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Operation complexity analysis
#[derive(Debug, Clone)]
pub struct OperationComplexity {
    pub operation_type: String,
    pub average_flops: u64,
    pub average_memory_bytes: usize,
    pub parallelization_factor: f64,
    pub gpu_acceleration_factor: f64,
}

impl OperationComplexity {
    /// Create complexity analysis for a given operation type
    pub fn new(operation_type: String) -> Self {
        let (avg_flops, avg_memory, parallel_factor, gpu_factor) =
            Self::get_default_complexity(&operation_type);

        Self {
            operation_type,
            average_flops: avg_flops,
            average_memory_bytes: avg_memory,
            parallelization_factor: parallel_factor,
            gpu_acceleration_factor: gpu_factor,
        }
    }

    /// Get default complexity estimates for common operation types
    fn get_default_complexity(op_type: &str) -> (u64, usize, f64, f64) {
        match op_type {
            "conv2d" => (1_000_000, 10_240, 0.8, 10.0),
            "conv3d" => (10_000_000, 102_400, 0.9, 15.0),
            "matmul" => (100_000, 4_096, 0.95, 20.0),
            "relu" | "sigmoid" | "tanh" => (1_000, 1_024, 0.99, 5.0),
            "batch_norm" => (10_000, 2_048, 0.9, 8.0),
            "pool2d" => (50_000, 4_096, 0.85, 6.0),
            "softmax" => (10_000, 2_048, 0.7, 4.0),
            _ => (10_000, 4_096, 0.5, 2.0), // Default values
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subgraph_creation() {
        let mut subgraph = Subgraph::new("test_subgraph".to_string());
        assert_eq!(subgraph.id, "test_subgraph");
        assert_eq!(subgraph.operations.len(), 0);

        let operation = SubgraphOperation::new(
            "op1".to_string(),
            "conv2d".to_string(),
            vec!["input".to_string()],
            vec!["output".to_string()],
        )
        .with_estimates(1000, 2048);

        subgraph.add_operation(operation);
        assert_eq!(subgraph.operations.len(), 1);
        assert_eq!(subgraph.estimated_compute_cost, 1000.0);
    }

    #[test]
    fn test_tensor_size_calculation() {
        let tensor = SubgraphTensor::new(
            "test_tensor".to_string(),
            vec![32, 3, 224, 224],
            "f32".to_string(),
        );
        assert_eq!(tensor.size_bytes, 32 * 3 * 224 * 224 * 4);
    }

    #[test]
    fn test_communication_cost() {
        let tensor = SubgraphTensor::new("comm_tensor".to_string(), vec![1024], "f32".to_string());
        let mut comm =
            SubgraphCommunication::new("subgraph1".to_string(), "subgraph2".to_string(), tensor);

        let normal_cost = comm.communication_cost();
        comm.set_critical_path(true);
        let critical_cost = comm.communication_cost();

        assert_eq!(critical_cost, normal_cost * 2.0);
    }

    #[test]
    fn test_load_balance_calculation() {
        let mut result = SubgraphExtractionResult::new();

        let mut sg1 = Subgraph::new("sg1".to_string());
        sg1.estimated_compute_cost = 100.0;
        let mut sg2 = Subgraph::new("sg2".to_string());
        sg2.estimated_compute_cost = 200.0;

        result.add_subgraph(sg1);
        result.add_subgraph(sg2);
        result.calculate_load_balance();

        assert_eq!(result.load_balance_score, 0.5);
    }
}
