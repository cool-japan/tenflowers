//! Subgraph Extraction and Parallelization
//!
//! This module provides algorithms for extracting subgraphs from computation graphs
//! to enable parallel execution, distributed training, and pipeline parallelism.
//!
//! The module is organized as follows:
//! - `config`: Configuration and strategy definitions
//! - `types`: Core data structures for subgraphs, operations, and results
//! - `extractor`: Main extraction implementation with various algorithms
//!
//! # Examples
//!
//! ```
//! use tenflowers_autograd::subgraph_extraction::{
//!     SubgraphExtractor, SubgraphConfig, ExtractionStrategy
//! };
//!
//! // Create a configuration for pipeline parallelism
//! let config = SubgraphConfig::for_pipeline_parallelism(4);
//! let extractor = SubgraphExtractor::new(config);
//!
//! // Extract subgraphs from operations
//! // let result = extractor.extract_subgraphs(&operations)?;
//! ```

pub mod config;
pub mod extractor;
pub mod types;

// Re-export main types for convenience
pub use config::{ExtractionStrategy, SubgraphConfig};
pub use extractor::SubgraphExtractor;
pub use types::{
    GraphOperation, OperationComplexity, Subgraph, SubgraphCommunication, SubgraphExtractionResult,
    SubgraphOperation, SubgraphTensor,
};

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_end_to_end_extraction() {
        // Test a complete extraction workflow
        let config = SubgraphConfig::default();
        let extractor = SubgraphExtractor::new(config);

        let operations = vec![
            GraphOperation::<f32>::new(
                "input_op".to_string(),
                "Input".to_string(),
                vec![],
                vec!["data".to_string()],
            ),
            GraphOperation::<f32>::new(
                "conv_op".to_string(),
                "Conv2D".to_string(),
                vec!["data".to_string()],
                vec!["conv_out".to_string()],
            ),
            GraphOperation::<f32>::new(
                "relu_op".to_string(),
                "ReLU".to_string(),
                vec!["conv_out".to_string()],
                vec!["relu_out".to_string()],
            ),
            GraphOperation::<f32>::new(
                "pool_op".to_string(),
                "MaxPool2D".to_string(),
                vec!["relu_out".to_string()],
                vec!["pool_out".to_string()],
            ),
        ];

        let result = extractor.extract_subgraphs(&operations);
        assert!(result.is_ok());

        let extraction_result = result.unwrap();
        assert!(!extraction_result.subgraphs.is_empty());
        assert!(extraction_result.load_balance_score >= 0.0);
        assert!(extraction_result.parallel_efficiency >= 0.0);
    }

    #[test]
    fn test_different_strategies() {
        let strategies = vec![
            ExtractionStrategy::ByOperationType,
            ExtractionStrategy::ByComplexity,
            ExtractionStrategy::MinimalCommunication,
            ExtractionStrategy::Pipeline,
            ExtractionStrategy::DataParallel,
        ];

        let operations = vec![
            GraphOperation::<f32>::new(
                "op1".to_string(),
                "Conv2D".to_string(),
                vec!["input".to_string()],
                vec!["out1".to_string()],
            ),
            GraphOperation::<f32>::new(
                "op2".to_string(),
                "Conv2D".to_string(),
                vec!["out1".to_string()],
                vec!["out2".to_string()],
            ),
            GraphOperation::<f32>::new(
                "op3".to_string(),
                "ReLU".to_string(),
                vec!["out2".to_string()],
                vec!["out3".to_string()],
            ),
        ];

        for strategy in strategies {
            let config = SubgraphConfig {
                strategy,
                ..Default::default()
            };
            let extractor = SubgraphExtractor::new(config);

            let result = extractor.extract_subgraphs(&operations);
            assert!(result.is_ok(), "Strategy {:?} failed", strategy);
        }
    }

    #[test]
    fn test_config_factories() {
        let pipeline_config = SubgraphConfig::for_pipeline_parallelism(4);
        assert_eq!(pipeline_config.strategy, ExtractionStrategy::Pipeline);
        assert_eq!(pipeline_config.max_subgraphs, 4);

        let data_parallel_config = SubgraphConfig::for_data_parallelism(8);
        assert_eq!(
            data_parallel_config.strategy,
            ExtractionStrategy::DataParallel
        );
        assert_eq!(data_parallel_config.max_subgraphs, 8);

        let minimal_comm_config = SubgraphConfig::for_minimal_communication();
        assert_eq!(
            minimal_comm_config.strategy,
            ExtractionStrategy::MinimalCommunication
        );
        assert_eq!(minimal_comm_config.communication_cost_weight, 5.0);
    }
}
