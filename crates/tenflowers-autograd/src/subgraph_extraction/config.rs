//! Subgraph Extraction Configuration
//!
//! This module defines configuration options and strategies for subgraph extraction
//! from computational graphs to enable various forms of parallelism.

/// Subgraph extraction strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtractionStrategy {
    /// Split by operation types (e.g., all convolutions together)
    ByOperationType,
    /// Split by computational complexity
    ByComplexity,
    /// Split to minimize communication between subgraphs
    MinimalCommunication,
    /// Split for pipeline parallelism (sequential stages)
    Pipeline,
    /// Split for data parallelism (replicated subgraphs)
    DataParallel,
    /// Custom user-defined partitioning
    Custom,
}

/// Subgraph extraction configuration
#[derive(Debug, Clone)]
pub struct SubgraphConfig {
    pub strategy: ExtractionStrategy,
    pub max_subgraphs: usize,
    pub min_operations_per_subgraph: usize,
    pub max_operations_per_subgraph: usize,
    pub communication_cost_weight: f64,
    pub load_balance_weight: f64,
    pub prefer_sequential_operations: bool,
    pub enable_subgraph_fusion: bool,
}

impl Default for SubgraphConfig {
    fn default() -> Self {
        Self {
            strategy: ExtractionStrategy::ByComplexity,
            max_subgraphs: 8,
            min_operations_per_subgraph: 2,
            max_operations_per_subgraph: 50,
            communication_cost_weight: 2.0,
            load_balance_weight: 1.5,
            prefer_sequential_operations: true,
            enable_subgraph_fusion: true,
        }
    }
}

impl SubgraphConfig {
    /// Create a configuration optimized for pipeline parallelism
    pub fn for_pipeline_parallelism(num_stages: usize) -> Self {
        Self {
            strategy: ExtractionStrategy::Pipeline,
            max_subgraphs: num_stages,
            min_operations_per_subgraph: 1,
            max_operations_per_subgraph: usize::MAX,
            communication_cost_weight: 1.0,
            load_balance_weight: 2.0,
            prefer_sequential_operations: true,
            enable_subgraph_fusion: false,
        }
    }

    /// Create a configuration optimized for data parallelism
    pub fn for_data_parallelism(num_replicas: usize) -> Self {
        Self {
            strategy: ExtractionStrategy::DataParallel,
            max_subgraphs: num_replicas,
            min_operations_per_subgraph: 1,
            max_operations_per_subgraph: usize::MAX,
            communication_cost_weight: 0.5,
            load_balance_weight: 1.0,
            prefer_sequential_operations: false,
            enable_subgraph_fusion: true,
        }
    }

    /// Create a configuration optimized for minimal communication
    pub fn for_minimal_communication() -> Self {
        Self {
            strategy: ExtractionStrategy::MinimalCommunication,
            max_subgraphs: 4,
            min_operations_per_subgraph: 3,
            max_operations_per_subgraph: 100,
            communication_cost_weight: 5.0,
            load_balance_weight: 1.0,
            prefer_sequential_operations: false,
            enable_subgraph_fusion: true,
        }
    }

    /// Create a configuration based on operation types
    pub fn for_operation_grouping() -> Self {
        Self {
            strategy: ExtractionStrategy::ByOperationType,
            max_subgraphs: 16,
            min_operations_per_subgraph: 1,
            max_operations_per_subgraph: 200,
            communication_cost_weight: 1.0,
            load_balance_weight: 1.5,
            prefer_sequential_operations: false,
            enable_subgraph_fusion: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = SubgraphConfig::default();
        assert_eq!(config.strategy, ExtractionStrategy::ByComplexity);
        assert_eq!(config.max_subgraphs, 8);
        assert!(config.enable_subgraph_fusion);
    }

    #[test]
    fn test_pipeline_config() {
        let config = SubgraphConfig::for_pipeline_parallelism(4);
        assert_eq!(config.strategy, ExtractionStrategy::Pipeline);
        assert_eq!(config.max_subgraphs, 4);
        assert!(!config.enable_subgraph_fusion);
    }

    #[test]
    fn test_data_parallel_config() {
        let config = SubgraphConfig::for_data_parallelism(8);
        assert_eq!(config.strategy, ExtractionStrategy::DataParallel);
        assert_eq!(config.max_subgraphs, 8);
        assert_eq!(config.communication_cost_weight, 0.5);
    }

    #[test]
    fn test_minimal_communication_config() {
        let config = SubgraphConfig::for_minimal_communication();
        assert_eq!(config.strategy, ExtractionStrategy::MinimalCommunication);
        assert_eq!(config.communication_cost_weight, 5.0);
    }
}
