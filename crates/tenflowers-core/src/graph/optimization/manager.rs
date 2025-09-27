//! Graph optimization manager and statistics
//!
//! This module provides the main graph optimizer that orchestrates all optimization
//! passes and collects statistics about the optimization process.

use super::fusion::OperationFusionPass;
use super::memory::MemoryOptimizationPass;
use super::passes::OptimizationPass;
use super::passes::{CSEPass, ConstantFoldingPass, DeadCodeEliminationPass};
use super::placement::DevicePlacementOptimizationPass;
use crate::graph::Graph;
use crate::Result;
use std::collections::HashMap;

/// Graph optimization manager
pub struct GraphOptimizer {
    passes: Vec<Box<dyn OptimizationPass>>,
    max_iterations: usize,
}

impl GraphOptimizer {
    /// Create a new graph optimizer with default passes
    pub fn new() -> Self {
        let mut optimizer = Self::empty();
        optimizer.add_default_passes();
        optimizer
    }

    /// Create an empty optimizer with no passes
    pub fn empty() -> Self {
        Self {
            passes: Vec::new(),
            max_iterations: 10,
        }
    }

    /// Add default optimization passes
    pub fn add_default_passes(&mut self) {
        self.add_pass(Box::new(DeadCodeEliminationPass::new()));
        self.add_pass(Box::new(ConstantFoldingPass::new()));
        self.add_pass(Box::new(CSEPass::new()));
        self.add_pass(Box::new(OperationFusionPass::new()));
        self.add_pass(Box::new(MemoryOptimizationPass::new()));
        self.add_pass(Box::new(DevicePlacementOptimizationPass::new()));
    }

    /// Add an optimization pass
    pub fn add_pass(&mut self, pass: Box<dyn OptimizationPass>) {
        self.passes.push(pass);
        // Sort by priority (highest first)
        self.passes.sort_by_key(|b| std::cmp::Reverse(b.priority()));
    }

    /// Set maximum number of optimization iterations
    pub fn set_max_iterations(&mut self, max_iterations: usize) {
        self.max_iterations = max_iterations;
    }

    /// Get the number of optimization passes
    pub fn pass_count(&self) -> usize {
        self.passes.len()
    }

    /// Optimize a computation graph
    pub fn optimize(&self, graph: &mut Graph) -> Result<OptimizationStats> {
        let mut stats = OptimizationStats::new();
        let mut iteration = 0;

        while iteration < self.max_iterations {
            let mut changed_in_iteration = false;

            for pass in &self.passes {
                if pass.is_applicable(graph) {
                    let start_time = std::time::Instant::now();
                    let changed = pass.apply(graph)?;
                    let duration = start_time.elapsed();

                    stats.record_pass(pass.name(), duration, changed);

                    if changed {
                        changed_in_iteration = true;
                    }
                }
            }

            if !changed_in_iteration {
                break; // Converged
            }

            iteration += 1;
            stats.iterations += 1;
        }

        Ok(stats)
    }
}

impl Default for GraphOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about graph optimization
#[derive(Debug, Clone)]
pub struct OptimizationStats {
    pub iterations: usize,
    pub total_time: std::time::Duration,
    pub pass_stats: HashMap<String, PassStats>,
}

#[derive(Debug, Clone)]
pub struct PassStats {
    pub runs: usize,
    pub changes: usize,
    pub total_time: std::time::Duration,
}

impl OptimizationStats {
    fn new() -> Self {
        Self {
            iterations: 0,
            total_time: std::time::Duration::new(0, 0),
            pass_stats: HashMap::new(),
        }
    }

    fn record_pass(&mut self, pass_name: &str, duration: std::time::Duration, changed: bool) {
        self.total_time += duration;

        let stats = self
            .pass_stats
            .entry(pass_name.to_string())
            .or_insert(PassStats {
                runs: 0,
                changes: 0,
                total_time: std::time::Duration::new(0, 0),
            });

        stats.runs += 1;
        stats.total_time += duration;
        if changed {
            stats.changes += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_optimizer_creation() {
        let optimizer = GraphOptimizer::new();
        assert_eq!(optimizer.passes.len(), 6); // All default passes

        let empty_optimizer = GraphOptimizer::empty();
        assert_eq!(empty_optimizer.passes.len(), 0);
    }

    #[test]
    fn test_optimization_stats() {
        let mut stats = OptimizationStats::new();
        assert_eq!(stats.iterations, 0);
        assert_eq!(stats.pass_stats.len(), 0);

        stats.record_pass("test_pass", std::time::Duration::from_millis(10), true);
        assert_eq!(stats.pass_stats.len(), 1);
        assert!(stats.pass_stats.contains_key("test_pass"));
    }

    #[test]
    fn test_pass_addition() {
        let mut optimizer = GraphOptimizer::empty();

        // Add a pass and verify it's added
        optimizer.add_pass(Box::new(ConstantFoldingPass::new()));
        assert_eq!(optimizer.passes.len(), 1);

        // Add another pass and verify sorting by priority
        optimizer.add_pass(Box::new(DeadCodeEliminationPass::new()));
        assert_eq!(optimizer.passes.len(), 2);

        // Verify higher priority pass comes first
        assert!(optimizer.passes[0].priority() >= optimizer.passes[1].priority());
    }

    #[test]
    fn test_max_iterations() {
        let mut optimizer = GraphOptimizer::new();
        optimizer.set_max_iterations(5);
        assert_eq!(optimizer.max_iterations, 5);
    }

    #[test]
    fn test_pass_stats_recording() {
        let mut stats = OptimizationStats::new();

        // Record some passes
        stats.record_pass("pass1", std::time::Duration::from_millis(10), true);
        stats.record_pass("pass1", std::time::Duration::from_millis(5), false);
        stats.record_pass("pass2", std::time::Duration::from_millis(20), true);

        // Verify stats
        assert_eq!(stats.pass_stats.len(), 2);

        let pass1_stats = &stats.pass_stats["pass1"];
        assert_eq!(pass1_stats.runs, 2);
        assert_eq!(pass1_stats.changes, 1);
        assert_eq!(pass1_stats.total_time, std::time::Duration::from_millis(15));

        let pass2_stats = &stats.pass_stats["pass2"];
        assert_eq!(pass2_stats.runs, 1);
        assert_eq!(pass2_stats.changes, 1);
        assert_eq!(pass2_stats.total_time, std::time::Duration::from_millis(20));

        // Verify total time
        assert_eq!(stats.total_time, std::time::Duration::from_millis(35));
    }
}
