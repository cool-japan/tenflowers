//! Graph optimization framework
//!
//! This module provides a comprehensive optimization framework for computational graphs,
//! including various optimization passes, fusion patterns, memory optimization,
//! device placement strategies, and performance statistics.
//!
//! ## Module Structure
//!
//! - [`passes`]: Core optimization pass trait and basic passes (constant folding, CSE, dead code elimination)
//! - [`fusion`]: Operation fusion patterns and algorithms for kernel optimization
//! - [`memory`]: Memory optimization including tensor lifetime analysis and in-place operations
//! - [`placement`]: Device placement optimization strategies for multi-GPU systems
//! - [`manager`]: Graph optimizer orchestration and performance statistics

pub mod fusion;
pub mod manager;
pub mod memory;
pub mod passes;
pub mod placement;

// Re-export all public types for backward compatibility

// Core trait
pub use passes::OptimizationPass;

// Basic optimization passes
pub use passes::{
    AlgebraicSimplificationPass, CSEPass, ConstantFoldingPass, DeadCodeEliminationPass,
    OperationSchedulingPass, StrengthReductionPass,
};

// Operation fusion
pub use fusion::{FusionCandidate, FusionPattern, OperationFusionPass};

// Memory optimization
pub use memory::MemoryOptimizationPass;

// Device placement optimization
pub use placement::{DevicePlacementOptimizationPass, OperationProfile, PlacementStrategy};

// Graph optimizer and statistics
pub use manager::{GraphOptimizer, OptimizationStats, PassStats};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backward_compatibility_imports() {
        // Test that all major types can be imported directly from the optimization module

        // Core trait
        let _: Option<&dyn OptimizationPass> = None;

        // Basic passes
        let _constant_pass = ConstantFoldingPass::new();
        let _cse_pass = CSEPass::new();
        let _dce_pass = DeadCodeEliminationPass::new();

        // Advanced passes
        let _algebraic_pass = AlgebraicSimplificationPass::new();
        let _scheduling_pass = OperationSchedulingPass::new();
        let _strength_pass = StrengthReductionPass::new();

        // Fusion
        let _fusion_pass = OperationFusionPass::new();
        let _pattern = FusionPattern::MatMulAdd;

        // Memory optimization
        let _memory_pass = MemoryOptimizationPass::new();

        // Device placement
        let _placement_pass = DevicePlacementOptimizationPass::new();
        let _strategy = PlacementStrategy::Hybrid;

        // Manager and stats
        let _optimizer = GraphOptimizer::new();
        let _stats = OptimizationStats {
            iterations: 0,
            total_time: std::time::Duration::new(0, 0),
            pass_stats: std::collections::HashMap::new(),
        };
    }

    #[test]
    fn test_optimization_pass_trait_usage() {
        // Test that the trait can be used polymorphically
        let passes: Vec<Box<dyn OptimizationPass>> = vec![
            Box::new(ConstantFoldingPass::new()),
            Box::new(CSEPass::new()),
            Box::new(DeadCodeEliminationPass::new()),
            Box::new(AlgebraicSimplificationPass::new()),
            Box::new(OperationSchedulingPass::new()),
            Box::new(StrengthReductionPass::new()),
            Box::new(OperationFusionPass::new()),
            Box::new(MemoryOptimizationPass::new()),
            Box::new(DevicePlacementOptimizationPass::new()),
        ];

        // Verify all passes have expected properties
        for pass in &passes {
            assert!(!pass.name().is_empty());
            assert!(pass.priority() <= 1000); // Reasonable priority range
        }

        // Verify they're sorted by priority correctly when collected
        let mut priorities: Vec<u32> = passes.iter().map(|p| p.priority()).collect();
        priorities.sort_by(|a, b| b.cmp(a)); // Sort descending (highest first)

        // Check that higher priority passes have higher values
        for window in priorities.windows(2) {
            assert!(window[0] >= window[1]);
        }
    }

    #[test]
    fn test_graph_optimizer_with_all_passes() {
        // Test that GraphOptimizer can be created and configured with all passes
        let mut optimizer = GraphOptimizer::empty();

        // Add all types of passes
        optimizer.add_pass(Box::new(ConstantFoldingPass::new()));
        optimizer.add_pass(Box::new(CSEPass::new()));
        optimizer.add_pass(Box::new(DeadCodeEliminationPass::new()));
        optimizer.add_pass(Box::new(AlgebraicSimplificationPass::new()));
        optimizer.add_pass(Box::new(OperationSchedulingPass::new()));
        optimizer.add_pass(Box::new(StrengthReductionPass::new()));
        optimizer.add_pass(Box::new(OperationFusionPass::new()));
        optimizer.add_pass(Box::new(MemoryOptimizationPass::new()));
        optimizer.add_pass(Box::new(DevicePlacementOptimizationPass::new()));

        // Verify all passes were added
        assert_eq!(optimizer.pass_count(), 9);

        // Test default optimizer
        let default_optimizer = GraphOptimizer::new();
        assert_eq!(default_optimizer.pass_count(), 9);
    }

    #[test]
    fn test_fusion_patterns_and_candidates() {
        // Test fusion pattern variants
        let patterns = vec![
            FusionPattern::MatMulAdd,
            FusionPattern::AddActivation,
            FusionPattern::ConvBatchNormReLU,
        ];

        for pattern in patterns {
            let candidate = FusionCandidate {
                pattern: pattern.clone(),
                nodes: vec![1, 2, 3],
            };
            assert_eq!(candidate.nodes.len(), 3);
        }
    }

    #[test]
    fn test_placement_strategies() {
        // Test all placement strategy variants
        let strategies = vec![
            PlacementStrategy::MinimizeCommunication,
            PlacementStrategy::LoadBalancing,
            PlacementStrategy::MemoryOptimized,
            PlacementStrategy::Hybrid,
        ];

        for strategy in strategies {
            let pass = DevicePlacementOptimizationPass::new().with_strategy(strategy);
            assert_eq!(pass.name(), "DevicePlacementOptimization");
        }
    }

    #[test]
    fn test_operation_profile_creation() {
        // Test that OperationProfile can be created with all fields
        let profile = OperationProfile {
            compute_intensity: 5.0,
            memory_usage: 1024 * 1024, // 1MB
            parallelizable: true,
            gpu_optimized: true,
            communication_cost: 2.0,
        };

        assert_eq!(profile.compute_intensity, 5.0);
        assert_eq!(profile.memory_usage, 1024 * 1024);
        assert!(profile.parallelizable);
        assert!(profile.gpu_optimized);
        assert_eq!(profile.communication_cost, 2.0);
    }
}
