//! Subgraph Extraction Implementation
//!
//! This module contains the main SubgraphExtractor implementation with various
//! extraction strategies for computational graph partitioning.

use super::config::{ExtractionStrategy, SubgraphConfig};
use super::types::{
    GraphOperation, OperationComplexity, Subgraph, SubgraphExtractionResult, SubgraphOperation,
};
use crate::Result;
use std::collections::{HashMap, HashSet, VecDeque};
use tenflowers_core::TensorError;

/// Subgraph extraction optimizer
#[derive(Debug)]
pub struct SubgraphExtractor {
    config: SubgraphConfig,
    operation_profiles: HashMap<String, OperationComplexity>,
}

impl SubgraphExtractor {
    /// Create a new subgraph extractor
    pub fn new(config: SubgraphConfig) -> Self {
        let mut extractor = Self {
            config,
            operation_profiles: HashMap::new(),
        };

        extractor.initialize_operation_profiles();
        extractor
    }

    /// Initialize operation complexity profiles
    fn initialize_operation_profiles(&mut self) {
        let profiles = vec![
            // Linear algebra - high compute, moderate communication
            OperationComplexity::new("MatMul".to_string()),
            OperationComplexity::new("Conv2D".to_string()),
            // Element-wise operations - low compute, high parallelization
            OperationComplexity::new("Add".to_string()),
            OperationComplexity::new("Mul".to_string()),
            // Activation functions
            OperationComplexity::new("ReLU".to_string()),
            OperationComplexity::new("Sigmoid".to_string()),
            // Reduction operations - moderate compute, synchronization needed
            OperationComplexity::new("Sum".to_string()),
            OperationComplexity::new("Mean".to_string()),
            // Normalization operations - complex dependencies
            OperationComplexity::new("BatchNorm".to_string()),
            // Memory operations - low compute, shape changes
            OperationComplexity::new("Reshape".to_string()),
            OperationComplexity::new("Transpose".to_string()),
        ];

        for profile in profiles {
            self.operation_profiles
                .insert(profile.operation_type.clone(), profile);
        }
    }

    /// Extract subgraphs from a computation graph
    pub fn extract_subgraphs<T>(
        &self,
        operations: &[GraphOperation<T>],
    ) -> Result<SubgraphExtractionResult>
    where
        T: Clone + std::fmt::Debug + 'static,
    {
        match self.config.strategy {
            ExtractionStrategy::ByOperationType => self.extract_by_operation_type(operations),
            ExtractionStrategy::ByComplexity => self.extract_by_complexity(operations),
            ExtractionStrategy::MinimalCommunication => {
                self.extract_minimal_communication(operations)
            }
            ExtractionStrategy::Pipeline => self.extract_pipeline(operations),
            ExtractionStrategy::DataParallel => self.extract_data_parallel(operations),
            ExtractionStrategy::Custom => self.extract_custom(operations),
        }
    }

    /// Extract subgraphs by grouping similar operation types
    fn extract_by_operation_type<T>(
        &self,
        operations: &[GraphOperation<T>],
    ) -> Result<SubgraphExtractionResult>
    where
        T: Clone + std::fmt::Debug + 'static,
    {
        let mut operation_groups: HashMap<String, Vec<&GraphOperation<T>>> = HashMap::new();

        // Group operations by type
        for op in operations {
            operation_groups
                .entry(op.operation_name.clone())
                .or_default()
                .push(op);
        }

        let mut subgraphs = Vec::new();
        let mut subgraph_id = 0;

        for (op_type, ops) in operation_groups {
            if ops.len() >= self.config.min_operations_per_subgraph {
                let subgraph = self.create_subgraph_from_operations(
                    format!("subgraph_{op_type}_{subgraph_id}"),
                    ops,
                )?;
                subgraphs.push(subgraph);
                subgraph_id += 1;
            }
        }

        self.finalize_extraction_result(subgraphs, operations)
    }

    /// Extract subgraphs by computational complexity
    fn extract_by_complexity<T>(
        &self,
        operations: &[GraphOperation<T>],
    ) -> Result<SubgraphExtractionResult>
    where
        T: Clone + std::fmt::Debug + 'static,
    {
        // Calculate complexity for each operation
        let mut operation_complexities: Vec<(usize, f64)> = operations
            .iter()
            .enumerate()
            .map(|(i, op)| (i, self.calculate_operation_complexity(op)))
            .collect();

        // Sort by complexity (descending)
        operation_complexities
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut subgraphs = Vec::new();
        let mut remaining_ops: HashSet<usize> = (0..operations.len()).collect();
        let mut subgraph_id = 0;

        while !remaining_ops.is_empty() && subgraphs.len() < self.config.max_subgraphs {
            let mut current_subgraph_ops = Vec::new();
            let mut current_complexity = 0.0;
            let target_complexity =
                self.calculate_target_complexity(&operation_complexities, &remaining_ops);

            // Greedily add operations to current subgraph
            for &(op_idx, complexity) in &operation_complexities {
                if remaining_ops.contains(&op_idx)
                    && current_subgraph_ops.len() < self.config.max_operations_per_subgraph
                    && (current_complexity + complexity <= target_complexity * 1.2
                        || current_subgraph_ops.is_empty())
                {
                    current_subgraph_ops.push(&operations[op_idx]);
                    current_complexity += complexity;
                    remaining_ops.remove(&op_idx);
                }
            }

            if current_subgraph_ops.len() >= self.config.min_operations_per_subgraph {
                let subgraph = self.create_subgraph_from_operations(
                    format!("complexity_subgraph_{subgraph_id}"),
                    current_subgraph_ops,
                )?;
                subgraphs.push(subgraph);
                subgraph_id += 1;
            } else {
                break; // Can't form more valid subgraphs
            }
        }

        self.finalize_extraction_result(subgraphs, operations)
    }

    /// Extract subgraphs to minimize communication
    fn extract_minimal_communication<T>(
        &self,
        operations: &[GraphOperation<T>],
    ) -> Result<SubgraphExtractionResult>
    where
        T: Clone + std::fmt::Debug + 'static,
    {
        // Build adjacency graph based on data dependencies
        let adjacency = self.build_adjacency_graph(operations);

        // Use graph clustering algorithm
        let clusters = self.cluster_operations_by_communication(operations, &adjacency)?;

        let mut subgraphs = Vec::new();
        for (cluster_id, cluster_ops) in clusters.iter().enumerate() {
            if cluster_ops.len() >= self.config.min_operations_per_subgraph {
                let ops_refs: Vec<&GraphOperation<T>> =
                    cluster_ops.iter().map(|&idx| &operations[idx]).collect();

                let subgraph = self.create_subgraph_from_operations(
                    format!("comm_subgraph_{cluster_id}"),
                    ops_refs,
                )?;
                subgraphs.push(subgraph);
            }
        }

        self.finalize_extraction_result(subgraphs, operations)
    }

    /// Extract subgraphs for pipeline parallelism
    fn extract_pipeline<T>(
        &self,
        operations: &[GraphOperation<T>],
    ) -> Result<SubgraphExtractionResult>
    where
        T: Clone + std::fmt::Debug + 'static,
    {
        // Perform topological sort to get execution order
        let topo_order = self.topological_sort_operations(operations)?;

        // Divide into pipeline stages
        let stages_per_subgraph =
            (operations.len() + self.config.max_subgraphs - 1) / self.config.max_subgraphs;
        let mut subgraphs = Vec::new();

        for (stage_id, chunk) in topo_order.chunks(stages_per_subgraph).enumerate() {
            let ops_refs: Vec<&GraphOperation<T>> =
                chunk.iter().map(|&idx| &operations[idx]).collect();

            if ops_refs.len() >= self.config.min_operations_per_subgraph {
                let subgraph = self.create_subgraph_from_operations(
                    format!("pipeline_stage_{stage_id}"),
                    ops_refs,
                )?;
                subgraphs.push(subgraph);
            }
        }

        self.finalize_extraction_result(subgraphs, operations)
    }

    /// Extract subgraphs for data parallelism
    fn extract_data_parallel<T>(
        &self,
        operations: &[GraphOperation<T>],
    ) -> Result<SubgraphExtractionResult>
    where
        T: Clone + std::fmt::Debug + 'static,
    {
        // For data parallelism, we replicate the entire graph
        let mut subgraphs = Vec::new();

        for replica_id in 0..self.config.max_subgraphs {
            let all_ops: Vec<&GraphOperation<T>> = operations.iter().collect();
            let subgraph = self.create_subgraph_from_operations(
                format!("data_parallel_replica_{replica_id}"),
                all_ops,
            )?;
            subgraphs.push(subgraph);
        }

        self.finalize_extraction_result(subgraphs, operations)
    }

    /// Extract subgraphs using custom strategy (placeholder)
    fn extract_custom<T>(
        &self,
        operations: &[GraphOperation<T>],
    ) -> Result<SubgraphExtractionResult>
    where
        T: Clone + std::fmt::Debug + 'static,
    {
        // Default to complexity-based extraction for custom strategy
        self.extract_by_complexity(operations)
    }

    /// Calculate operation complexity
    fn calculate_operation_complexity<T>(&self, operation: &GraphOperation<T>) -> f64 {
        if let Some(profile) = self.operation_profiles.get(&operation.operation_name) {
            let tensor_size = operation.tensor_sizes.iter().product::<usize>() as f64;
            profile.average_flops as f64 * tensor_size / 1_000_000.0 // Normalize
        } else {
            // Default complexity for unknown operations
            operation.tensor_sizes.iter().product::<usize>() as f64 / 1_000.0
        }
    }

    /// Calculate target complexity for balanced partitioning
    fn calculate_target_complexity(
        &self,
        operation_complexities: &[(usize, f64)],
        remaining_ops: &HashSet<usize>,
    ) -> f64 {
        let total_remaining_complexity: f64 = operation_complexities
            .iter()
            .filter_map(|(idx, complexity)| {
                if remaining_ops.contains(idx) {
                    Some(*complexity)
                } else {
                    None
                }
            })
            .sum();

        let remaining_subgraphs = (remaining_ops.len() + self.config.min_operations_per_subgraph
            - 1)
            / self.config.min_operations_per_subgraph;

        total_remaining_complexity / remaining_subgraphs.max(1) as f64
    }

    /// Build adjacency graph for communication analysis
    fn build_adjacency_graph<T>(&self, operations: &[GraphOperation<T>]) -> Vec<Vec<usize>> {
        let mut adjacency = vec![vec![]; operations.len()];
        let mut output_to_op: HashMap<String, usize> = HashMap::new();

        // Build output -> operation mapping
        for (i, op) in operations.iter().enumerate() {
            for output in &op.outputs {
                output_to_op.insert(output.clone(), i);
            }
        }

        // Build adjacency based on data dependencies
        for (i, op) in operations.iter().enumerate() {
            for input in &op.inputs {
                if let Some(&producer_op) = output_to_op.get(input) {
                    if producer_op != i {
                        adjacency[producer_op].push(i);
                    }
                }
            }
        }

        adjacency
    }

    /// Cluster operations to minimize communication
    fn cluster_operations_by_communication<T>(
        &self,
        operations: &[GraphOperation<T>],
        adjacency: &[Vec<usize>],
    ) -> Result<Vec<Vec<usize>>> {
        // Simplified clustering: group operations with strong dependencies
        let mut clusters = Vec::new();
        let mut visited = vec![false; operations.len()];

        for start_op in 0..operations.len() {
            if !visited[start_op] {
                let mut cluster = Vec::new();
                let mut queue = VecDeque::new();
                queue.push_back(start_op);
                visited[start_op] = true;

                while let Some(op_idx) = queue.pop_front() {
                    cluster.push(op_idx);

                    // Add strongly connected neighbors
                    for &neighbor in &adjacency[op_idx] {
                        if !visited[neighbor]
                            && cluster.len() < self.config.max_operations_per_subgraph
                        {
                            visited[neighbor] = true;
                            queue.push_back(neighbor);
                        }
                    }
                }

                if cluster.len() >= self.config.min_operations_per_subgraph {
                    clusters.push(cluster);
                }
            }
        }

        Ok(clusters)
    }

    /// Perform topological sort on operations
    fn topological_sort_operations<T>(
        &self,
        operations: &[GraphOperation<T>],
    ) -> Result<Vec<usize>> {
        let adjacency = self.build_adjacency_graph(operations);
        let mut in_degree = vec![0; operations.len()];

        // Calculate in-degrees
        for adj_list in &adjacency {
            for &neighbor in adj_list {
                in_degree[neighbor] += 1;
            }
        }

        let mut queue = VecDeque::new();
        let mut result = Vec::new();

        // Add nodes with zero in-degree
        for (i, &degree) in in_degree.iter().enumerate() {
            if degree == 0 {
                queue.push_back(i);
            }
        }

        while let Some(current) = queue.pop_front() {
            result.push(current);

            for &neighbor in &adjacency[current] {
                in_degree[neighbor] -= 1;
                if in_degree[neighbor] == 0 {
                    queue.push_back(neighbor);
                }
            }
        }

        if result.len() != operations.len() {
            return Err(TensorError::invalid_operation_simple(
                "Cycle detected in computation graph - cannot perform topological sort".to_string(),
            ));
        }

        Ok(result)
    }

    /// Create a subgraph from a collection of operations
    fn create_subgraph_from_operations<T>(
        &self,
        subgraph_id: String,
        operations: Vec<&GraphOperation<T>>,
    ) -> Result<Subgraph> {
        let mut subgraph = Subgraph::new(subgraph_id);

        for op in operations {
            let subgraph_op = SubgraphOperation::new(
                op.id.clone(),
                op.operation_name.clone(),
                op.inputs.clone(),
                op.outputs.clone(),
            );
            subgraph.add_operation(subgraph_op);
        }

        Ok(subgraph)
    }

    /// Finalize the extraction result with analysis
    fn finalize_extraction_result<T>(
        &self,
        subgraphs: Vec<Subgraph>,
        _operations: &[GraphOperation<T>],
    ) -> Result<SubgraphExtractionResult> {
        let mut result = SubgraphExtractionResult::new();

        for subgraph in subgraphs {
            result.add_subgraph(subgraph);
        }

        result.calculate_load_balance();

        // Calculate efficiency metrics
        result.parallel_efficiency = self.calculate_parallel_efficiency(&result);
        result.memory_efficiency = self.calculate_memory_efficiency(&result);

        Ok(result)
    }

    /// Calculate parallel efficiency
    fn calculate_parallel_efficiency(&self, result: &SubgraphExtractionResult) -> f64 {
        if result.subgraphs.is_empty() {
            return 0.0;
        }

        // Simple model: efficiency based on load balance and subgraph count
        let ideal_parallelism = result.subgraphs.len() as f64;
        let load_factor = result.load_balance_score;

        (load_factor * ideal_parallelism) / (ideal_parallelism + 1.0)
    }

    /// Calculate memory efficiency
    fn calculate_memory_efficiency(&self, result: &SubgraphExtractionResult) -> f64 {
        if result.subgraphs.is_empty() {
            return 0.0;
        }

        // Simple model: efficiency based on communication cost vs computation cost
        let total_compute_cost = result.total_compute_cost();
        let communication_cost = result.total_communication_cost;

        if total_compute_cost > 0.0 {
            total_compute_cost / (total_compute_cost + communication_cost)
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::subgraph_extraction::config::ExtractionStrategy;

    #[test]
    fn test_extractor_creation() {
        let config = SubgraphConfig::default();
        let extractor = SubgraphExtractor::new(config);
        assert!(!extractor.operation_profiles.is_empty());
    }

    #[test]
    fn test_operation_complexity_calculation() {
        let config = SubgraphConfig::default();
        let extractor = SubgraphExtractor::new(config);

        let mut operation = GraphOperation::<f32>::new(
            "test_op".to_string(),
            "MatMul".to_string(),
            vec!["input1".to_string()],
            vec!["output1".to_string()],
        );
        operation.tensor_sizes = vec![1024, 1024];

        let complexity = extractor.calculate_operation_complexity(&operation);
        assert!(complexity > 0.0);
    }

    #[test]
    fn test_empty_operations_extraction() {
        let config = SubgraphConfig::default();
        let extractor = SubgraphExtractor::new(config);
        let operations: Vec<GraphOperation<f32>> = Vec::new();

        let result = extractor.extract_subgraphs(&operations);
        assert!(result.is_ok());

        let extraction_result = result.unwrap();
        assert_eq!(extraction_result.subgraphs.len(), 0);
    }

    #[test]
    fn test_pipeline_extraction() {
        let config = SubgraphConfig::for_pipeline_parallelism(2);
        let extractor = SubgraphExtractor::new(config);

        let operations = vec![
            GraphOperation::<f32>::new(
                "op1".to_string(),
                "Conv2D".to_string(),
                vec!["input".to_string()],
                vec!["intermediate1".to_string()],
            ),
            GraphOperation::<f32>::new(
                "op2".to_string(),
                "ReLU".to_string(),
                vec!["intermediate1".to_string()],
                vec!["intermediate2".to_string()],
            ),
            GraphOperation::<f32>::new(
                "op3".to_string(),
                "MatMul".to_string(),
                vec!["intermediate2".to_string()],
                vec!["output".to_string()],
            ),
        ];

        let result = extractor.extract_subgraphs(&operations);
        assert!(result.is_ok());

        let extraction_result = result.unwrap();
        assert!(!extraction_result.subgraphs.is_empty());
    }
}
