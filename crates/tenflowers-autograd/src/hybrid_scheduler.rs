//! # Hybrid Differentiation Scheduler
//!
//! This module provides an intelligent scheduler for automatic selection between
//! forward-mode and reverse-mode automatic differentiation, optimizing for both
//! computational efficiency and memory usage.
//!
//! ## Features
//!
//! - **Automatic Mode Selection**: Analyzes computation graphs to select optimal AD mode
//! - **Dynamic Scheduling**: Adapts strategy based on runtime performance metrics
//! - **Memory-Aware**: Considers memory constraints when making scheduling decisions
//! - **Cost Modeling**: Sophisticated cost estimation for different AD strategies
//! - **Performance Tracking**: Monitors execution and learns from past decisions
//!
//! ## Usage
//!
//! ```rust,no_run
//! use tenflowers_autograd::{HybridScheduler, SchedulerConfig, GradientTape};
//! use tenflowers_core::Tensor;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create scheduler with configuration
//! let config = SchedulerConfig::default()
//!     .with_memory_budget_mb(2048)
//!     .with_adaptive_learning(true);
//!
//! let mut scheduler = HybridScheduler::new(config);
//!
//! // Let scheduler decide optimal strategy
//! let tape = GradientTape::new();
//! let x = tape.watch(Tensor::<f32>::ones(&[100, 10]));
//! let y = tape.watch(Tensor::<f32>::ones(&[100, 1]));
//!
//! let strategy = scheduler.recommend_strategy(&y, &[&x])?;
//! println!("Recommended strategy: {:?}", strategy);
//! # Ok(())
//! # }
//! ```

use crate::forward_reverse::{
    ComplexityEstimate, DifferentiationMode, ForwardReverseDifferentiator,
};
use crate::{GradientTape, TrackedTensor};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tenflowers_core::{Result, Tensor, TensorError};

/// Configuration for the hybrid scheduler
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Memory budget in MB
    pub memory_budget_mb: usize,
    /// Enable adaptive learning from past executions
    pub adaptive_learning: bool,
    /// Threshold ratio for auto mode selection (input_dim / output_dim)
    pub mode_selection_threshold: f64,
    /// Weight for computation cost in decision making (0.0 to 1.0)
    pub compute_weight: f64,
    /// Weight for memory cost in decision making (0.0 to 1.0)
    pub memory_weight: f64,
    /// Enable graph analysis for optimization
    pub enable_graph_analysis: bool,
    /// Maximum graph depth to analyze
    pub max_graph_depth: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            memory_budget_mb: 2048, // 2GB default
            adaptive_learning: true,
            mode_selection_threshold: 1.0,
            compute_weight: 0.6,
            memory_weight: 0.4,
            enable_graph_analysis: true,
            max_graph_depth: 100,
        }
    }
}

impl SchedulerConfig {
    /// Set memory budget in MB
    pub fn with_memory_budget_mb(mut self, mb: usize) -> Self {
        self.memory_budget_mb = mb;
        self
    }

    /// Enable/disable adaptive learning
    pub fn with_adaptive_learning(mut self, enabled: bool) -> Self {
        self.adaptive_learning = enabled;
        self
    }

    /// Set mode selection threshold
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.mode_selection_threshold = threshold;
        self
    }

    /// Set computation vs memory weight (compute_weight + memory_weight should = 1.0)
    pub fn with_weights(mut self, compute: f64, memory: f64) -> Self {
        self.compute_weight = compute;
        self.memory_weight = memory;
        self
    }

    /// Enable graph analysis
    pub fn with_graph_analysis(mut self, enabled: bool) -> Self {
        self.enable_graph_analysis = enabled;
        self
    }
}

/// Execution statistics for a differentiation strategy
#[derive(Debug, Clone)]
pub struct ExecutionStats {
    /// Differentiation mode used
    pub mode: DifferentiationMode,
    /// Execution time
    pub execution_time: Duration,
    /// Peak memory usage (bytes)
    pub peak_memory_bytes: usize,
    /// Number of operations executed
    pub operation_count: usize,
    /// Input dimensions
    pub input_dims: Vec<usize>,
    /// Output dimensions
    pub output_dims: Vec<usize>,
}

/// Cost estimate for a differentiation strategy
#[derive(Debug, Clone)]
pub struct StrategyCost {
    /// Estimated computational cost (arbitrary units)
    pub compute_cost: f64,
    /// Estimated memory cost (bytes)
    pub memory_cost: usize,
    /// Combined weighted cost
    pub total_cost: f64,
    /// Recommended mode
    pub recommended_mode: DifferentiationMode,
}

/// Graph analysis result
#[derive(Debug, Clone)]
pub struct GraphAnalysis {
    /// Total number of operations in graph
    pub operation_count: usize,
    /// Graph depth (longest path)
    pub graph_depth: usize,
    /// Number of intermediate tensors
    pub intermediate_tensors: usize,
    /// Estimated memory footprint (bytes)
    pub estimated_memory: usize,
    /// Whether graph has loops/cycles
    pub has_cycles: bool,
    /// Parallelization potential (0.0 to 1.0)
    pub parallelization_potential: f64,
}

/// Hybrid differentiation scheduler
///
/// Intelligently selects between forward-mode and reverse-mode AD based on
/// problem characteristics, performance history, and resource constraints.
pub struct HybridScheduler {
    config: SchedulerConfig,
    differentiator: ForwardReverseDifferentiator,
    execution_history: Vec<ExecutionStats>,
    performance_cache: HashMap<String, StrategyCost>,
}

impl HybridScheduler {
    /// Create a new hybrid scheduler
    pub fn new(config: SchedulerConfig) -> Self {
        let differentiator = ForwardReverseDifferentiator::default();
        Self {
            config,
            differentiator,
            execution_history: Vec::new(),
            performance_cache: HashMap::new(),
        }
    }

    /// Recommend differentiation strategy based on problem characteristics
    pub fn recommend_strategy<T>(
        &mut self,
        target: &TrackedTensor<T>,
        inputs: &[&TrackedTensor<T>],
    ) -> Result<DifferentiationMode>
    where
        T: Clone + Default + Send + Sync + 'static,
    {
        // Step 1: Analyze problem dimensions
        let input_dims: Vec<usize> = inputs
            .iter()
            .map(|t| t.tensor.shape().dims().iter().product())
            .collect();
        let output_dims: Vec<usize> = vec![target.tensor.shape().dims().iter().product()];

        let total_input_dim: usize = input_dims.iter().sum();
        let total_output_dim: usize = output_dims.iter().sum();

        // Step 2: Check cache for similar problems
        let cache_key = self.generate_cache_key(&input_dims, &output_dims);
        if let Some(cached_cost) = self.performance_cache.get(&cache_key) {
            return Ok(cached_cost.recommended_mode);
        }

        // Step 3: Estimate costs for different strategies
        let strategy_cost = self.estimate_strategy_cost(total_input_dim, total_output_dim)?;

        // Step 4: Apply adaptive learning if enabled
        let recommended_mode =
            if self.config.adaptive_learning && !self.execution_history.is_empty() {
                self.adaptive_mode_selection(&strategy_cost, &input_dims, &output_dims)
            } else {
                strategy_cost.recommended_mode
            };

        // Step 5: Cache the decision
        self.performance_cache.insert(cache_key, strategy_cost);

        Ok(recommended_mode)
    }

    /// Estimate cost for different differentiation strategies
    pub fn estimate_strategy_cost(
        &self,
        input_dim: usize,
        output_dim: usize,
    ) -> Result<StrategyCost> {
        // Forward mode: Cost scales with input dimension
        // O(input_dim) for each output
        let forward_compute = (input_dim * output_dim) as f64;
        let forward_memory = input_dim * std::mem::size_of::<f64>();

        // Reverse mode: Cost scales with output dimension
        // O(output_dim) for each input
        let reverse_compute = (input_dim * output_dim) as f64;
        let reverse_memory = output_dim * std::mem::size_of::<f64>();

        // Apply weights
        let forward_cost = self.config.compute_weight * forward_compute
            + self.config.memory_weight * (forward_memory as f64);
        let reverse_cost = self.config.compute_weight * reverse_compute
            + self.config.memory_weight * (reverse_memory as f64);

        // Select mode based on cost
        let (recommended_mode, compute_cost, memory_cost, total_cost) =
            if forward_cost < reverse_cost {
                (
                    DifferentiationMode::Forward,
                    forward_compute,
                    forward_memory,
                    forward_cost,
                )
            } else {
                (
                    DifferentiationMode::Reverse,
                    reverse_compute,
                    reverse_memory,
                    reverse_cost,
                )
            };

        Ok(StrategyCost {
            compute_cost,
            memory_cost,
            total_cost,
            recommended_mode,
        })
    }

    /// Analyze computation graph structure
    pub fn analyze_graph<T>(&self, target: &TrackedTensor<T>) -> Result<GraphAnalysis>
    where
        T: Clone + Default + Send + Sync + 'static,
    {
        // Simplified graph analysis
        // In a full implementation, this would traverse the tape's computation graph

        let shape_product: usize = target.tensor.shape().dims().iter().product();

        Ok(GraphAnalysis {
            operation_count: 1, // Simplified
            graph_depth: 1,
            intermediate_tensors: 1,
            estimated_memory: shape_product * std::mem::size_of::<T>(),
            has_cycles: false,
            parallelization_potential: 0.5,
        })
    }

    /// Record execution statistics for adaptive learning
    pub fn record_execution(&mut self, stats: ExecutionStats) {
        self.execution_history.push(stats);

        // Keep history bounded
        if self.execution_history.len() > 1000 {
            self.execution_history.drain(0..500); // Keep most recent 500
        }
    }

    /// Get execution statistics summary
    pub fn get_execution_summary(&self) -> ExecutionSummary {
        if self.execution_history.is_empty() {
            return ExecutionSummary::default();
        }

        let total_executions = self.execution_history.len();
        let forward_count = self
            .execution_history
            .iter()
            .filter(|s| s.mode == DifferentiationMode::Forward)
            .count();
        let reverse_count = self
            .execution_history
            .iter()
            .filter(|s| s.mode == DifferentiationMode::Reverse)
            .count();

        let avg_forward_time = if forward_count > 0 {
            self.execution_history
                .iter()
                .filter(|s| s.mode == DifferentiationMode::Forward)
                .map(|s| s.execution_time.as_millis() as f64)
                .sum::<f64>()
                / forward_count as f64
        } else {
            0.0
        };

        let avg_reverse_time = if reverse_count > 0 {
            self.execution_history
                .iter()
                .filter(|s| s.mode == DifferentiationMode::Reverse)
                .map(|s| s.execution_time.as_millis() as f64)
                .sum::<f64>()
                / reverse_count as f64
        } else {
            0.0
        };

        ExecutionSummary {
            total_executions,
            forward_mode_count: forward_count,
            reverse_mode_count: reverse_count,
            avg_forward_time_ms: avg_forward_time,
            avg_reverse_time_ms: avg_reverse_time,
            cache_hit_rate: self.calculate_cache_hit_rate(),
        }
    }

    /// Check if memory budget allows for a strategy
    pub fn check_memory_budget(&self, estimated_memory_bytes: usize) -> bool {
        estimated_memory_bytes <= self.config.memory_budget_mb * 1024 * 1024
    }

    /// Clear execution history
    pub fn clear_history(&mut self) {
        self.execution_history.clear();
        self.performance_cache.clear();
    }

    /// Get current configuration
    pub fn get_config(&self) -> &SchedulerConfig {
        &self.config
    }

    /// Update configuration
    pub fn update_config(&mut self, config: SchedulerConfig) {
        self.config = config;
        // Clear cache when config changes
        self.performance_cache.clear();
    }

    // Private helper methods

    fn generate_cache_key(&self, input_dims: &[usize], output_dims: &[usize]) -> String {
        format!("in:{:?}_out:{:?}", input_dims, output_dims)
    }

    fn adaptive_mode_selection(
        &self,
        base_cost: &StrategyCost,
        input_dims: &[usize],
        output_dims: &[usize],
    ) -> DifferentiationMode {
        // Look for similar problems in history
        let similar_executions: Vec<&ExecutionStats> = self
            .execution_history
            .iter()
            .filter(|stats| {
                self.is_similar_problem(&stats.input_dims, input_dims)
                    && self.is_similar_problem(&stats.output_dims, output_dims)
            })
            .collect();

        if similar_executions.is_empty() {
            return base_cost.recommended_mode;
        }

        // Calculate average performance for each mode
        let forward_stats: Vec<_> = similar_executions
            .iter()
            .filter(|s| s.mode == DifferentiationMode::Forward)
            .collect();

        let reverse_stats: Vec<_> = similar_executions
            .iter()
            .filter(|s| s.mode == DifferentiationMode::Reverse)
            .collect();

        // If we have history for both modes, use the faster one
        if !forward_stats.is_empty() && !reverse_stats.is_empty() {
            let avg_forward = forward_stats
                .iter()
                .map(|s| s.execution_time.as_nanos() as f64)
                .sum::<f64>()
                / forward_stats.len() as f64;

            let avg_reverse = reverse_stats
                .iter()
                .map(|s| s.execution_time.as_nanos() as f64)
                .sum::<f64>()
                / reverse_stats.len() as f64;

            if avg_forward < avg_reverse {
                DifferentiationMode::Forward
            } else {
                DifferentiationMode::Reverse
            }
        } else {
            base_cost.recommended_mode
        }
    }

    fn is_similar_problem(&self, dims1: &[usize], dims2: &[usize]) -> bool {
        if dims1.len() != dims2.len() {
            return false;
        }

        // Allow 20% variation in dimensions
        dims1.iter().zip(dims2.iter()).all(|(d1, d2)| {
            let ratio = *d1 as f64 / *d2 as f64;
            (0.8..=1.2).contains(&ratio)
        })
    }

    fn calculate_cache_hit_rate(&self) -> f64 {
        if self.execution_history.is_empty() {
            return 0.0;
        }

        let total = self.execution_history.len();
        let cached_decisions = self
            .execution_history
            .iter()
            .filter(|stats| {
                let key = self.generate_cache_key(&stats.input_dims, &stats.output_dims);
                self.performance_cache.contains_key(&key)
            })
            .count();

        cached_decisions as f64 / total as f64
    }
}

impl Default for HybridScheduler {
    fn default() -> Self {
        Self::new(SchedulerConfig::default())
    }
}

/// Summary of execution statistics
#[derive(Debug, Clone, Default)]
pub struct ExecutionSummary {
    /// Total number of executions
    pub total_executions: usize,
    /// Number of forward mode executions
    pub forward_mode_count: usize,
    /// Number of reverse mode executions
    pub reverse_mode_count: usize,
    /// Average forward mode execution time (ms)
    pub avg_forward_time_ms: f64,
    /// Average reverse mode execution time (ms)
    pub avg_reverse_time_ms: f64,
    /// Cache hit rate (0.0 to 1.0)
    pub cache_hit_rate: f64,
}

impl ExecutionSummary {
    /// Print summary report
    pub fn print_report(&self) {
        println!("\n=== Hybrid Scheduler Execution Summary ===");
        println!("Total executions: {}", self.total_executions);
        println!(
            "Forward mode: {} ({:.1}%)",
            self.forward_mode_count,
            (self.forward_mode_count as f64 / self.total_executions as f64 * 100.0)
        );
        println!(
            "Reverse mode: {} ({:.1}%)",
            self.reverse_mode_count,
            (self.reverse_mode_count as f64 / self.total_executions as f64 * 100.0)
        );

        if self.forward_mode_count > 0 {
            println!("Avg forward time: {:.2} ms", self.avg_forward_time_ms);
        }
        if self.reverse_mode_count > 0 {
            println!("Avg reverse time: {:.2} ms", self.avg_reverse_time_ms);
        }

        println!("Cache hit rate: {:.1}%", self.cache_hit_rate * 100.0);
        println!("==========================================\n");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scheduler_config_builder() {
        let config = SchedulerConfig::default()
            .with_memory_budget_mb(4096)
            .with_adaptive_learning(false)
            .with_threshold(2.0)
            .with_weights(0.7, 0.3);

        assert_eq!(config.memory_budget_mb, 4096);
        assert!(!config.adaptive_learning);
        assert_eq!(config.mode_selection_threshold, 2.0);
        assert_eq!(config.compute_weight, 0.7);
        assert_eq!(config.memory_weight, 0.3);
    }

    #[test]
    fn test_scheduler_creation() {
        let scheduler = HybridScheduler::default();
        assert_eq!(scheduler.config.memory_budget_mb, 2048);
        assert!(scheduler.config.adaptive_learning);
    }

    #[test]
    fn test_strategy_cost_estimation() {
        let scheduler = HybridScheduler::default();

        // Many inputs, few outputs -> reverse mode cheaper
        let cost = scheduler.estimate_strategy_cost(100, 1).unwrap();
        assert_eq!(cost.recommended_mode, DifferentiationMode::Reverse);

        // Few inputs, many outputs -> forward mode cheaper
        let cost = scheduler.estimate_strategy_cost(1, 100).unwrap();
        assert_eq!(cost.recommended_mode, DifferentiationMode::Forward);
    }

    #[test]
    fn test_memory_budget_check() {
        let scheduler = HybridScheduler::default();

        // 1MB should fit in 2GB budget
        assert!(scheduler.check_memory_budget(1024 * 1024));

        // 10GB should not fit in 2GB budget
        assert!(!scheduler.check_memory_budget(10 * 1024 * 1024 * 1024));
    }

    #[test]
    fn test_execution_recording() {
        let mut scheduler = HybridScheduler::default();

        let stats = ExecutionStats {
            mode: DifferentiationMode::Forward,
            execution_time: Duration::from_millis(100),
            peak_memory_bytes: 1024,
            operation_count: 10,
            input_dims: vec![10],
            output_dims: vec![1],
        };

        scheduler.record_execution(stats);

        let summary = scheduler.get_execution_summary();
        assert_eq!(summary.total_executions, 1);
        assert_eq!(summary.forward_mode_count, 1);
        assert_eq!(summary.reverse_mode_count, 0);
    }

    #[test]
    fn test_cache_key_generation() {
        let scheduler = HybridScheduler::default();

        let key1 = scheduler.generate_cache_key(&[10, 20], &[1]);
        let key2 = scheduler.generate_cache_key(&[10, 20], &[1]);
        let key3 = scheduler.generate_cache_key(&[10, 30], &[1]);

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_execution_summary() {
        let mut scheduler = HybridScheduler::default();

        // Add multiple executions
        for i in 0..10 {
            let mode = if i % 2 == 0 {
                DifferentiationMode::Forward
            } else {
                DifferentiationMode::Reverse
            };

            scheduler.record_execution(ExecutionStats {
                mode,
                execution_time: Duration::from_millis(100 + i * 10),
                peak_memory_bytes: 1024,
                operation_count: 10,
                input_dims: vec![10],
                output_dims: vec![1],
            });
        }

        let summary = scheduler.get_execution_summary();
        assert_eq!(summary.total_executions, 10);
        assert_eq!(summary.forward_mode_count, 5);
        assert_eq!(summary.reverse_mode_count, 5);
    }

    #[test]
    fn test_clear_history() {
        let mut scheduler = HybridScheduler::default();

        scheduler.record_execution(ExecutionStats {
            mode: DifferentiationMode::Forward,
            execution_time: Duration::from_millis(100),
            peak_memory_bytes: 1024,
            operation_count: 10,
            input_dims: vec![10],
            output_dims: vec![1],
        });

        assert_eq!(scheduler.get_execution_summary().total_executions, 1);

        scheduler.clear_history();
        assert_eq!(scheduler.get_execution_summary().total_executions, 0);
    }
}
