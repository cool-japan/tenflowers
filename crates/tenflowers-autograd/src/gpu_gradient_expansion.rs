//! # GPU Gradient Expansion Strategy
//!
//! This module provides utilities and strategies for expanding GPU gradient support
//! across all autograd operations. It includes gap analysis, priority ranking,
//! and implementation scaffolding for GPU-accelerated backward passes.
//!
//! ## Overview
//!
//! GPU gradient computation can provide significant speedups for large-scale models,
//! but requires careful implementation to handle:
//! - Memory management (device transfers, buffers)
//! - Numerical stability (mixed precision considerations)
//! - Kernel optimization (fusion opportunities)
//! - Fallback strategies (CPU backup for unsupported ops)
//!
//! ## Usage
//!
//! ```rust,no_run
//! use tenflowers_autograd::gpu_gradient_expansion::{GpuGradientPlanner, Priority};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let planner = GpuGradientPlanner::new();
//!
//! // Analyze current GPU gradient coverage
//! let coverage = planner.analyze_coverage();
//! println!("GPU coverage: {}/{} operations", coverage.gpu_count, coverage.total_count);
//!
//! // Get implementation plan
//! let plan = planner.generate_implementation_plan(Priority::High);
//! for task in plan.tasks {
//!     println!("Implement GPU gradient for: {}", task.operation_name);
//! }
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;

/// Priority level for GPU gradient implementation
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Priority {
    /// Low priority / rarely used
    Low,
    /// Medium value
    Medium,
    /// High value for performance
    High,
    /// Critical for common deep learning workloads
    Critical,
}

/// GPU gradient implementation status
#[derive(Debug, Clone, PartialEq)]
pub enum GpuGradStatus {
    /// Fully implemented with optimized kernels
    Implemented,
    /// Partially implemented (e.g., only for specific dtypes)
    Partial,
    /// Not implemented yet
    Missing,
    /// Planned but not started
    Planned,
}

/// Information about a GPU gradient operation
#[derive(Debug, Clone)]
pub struct GpuGradInfo {
    /// Operation name
    pub operation_name: String,
    /// Category (e.g., "LinearAlgebra", "Activations")
    pub category: String,
    /// Current implementation status
    pub status: GpuGradStatus,
    /// Priority for implementation
    pub priority: Priority,
    /// Estimated speedup vs CPU (e.g., 10.0 = 10x faster)
    pub estimated_speedup: f64,
    /// Memory overhead (bytes per element)
    pub memory_overhead_bytes: usize,
    /// Implementation complexity (1-10 scale)
    pub complexity: u8,
    /// Dependencies on other GPU ops
    pub dependencies: Vec<String>,
    /// Notes and implementation hints
    pub notes: String,
}

/// Coverage analysis result
#[derive(Debug, Clone)]
pub struct GpuCoverageAnalysis {
    /// Total number of gradient operations
    pub total_count: usize,
    /// Number with GPU support
    pub gpu_count: usize,
    /// Coverage percentage
    pub coverage_percentage: f64,
    /// Operations by category
    pub by_category: HashMap<String, GpuCategoryCoverage>,
    /// Missing operations by priority
    pub missing_by_priority: HashMap<Priority, Vec<String>>,
}

/// Coverage for a specific category
#[derive(Debug, Clone)]
pub struct GpuCategoryCoverage {
    pub total: usize,
    pub gpu_supported: usize,
    pub percentage: f64,
}

/// Implementation plan for GPU gradients
#[derive(Debug, Clone)]
pub struct ImplementationPlan {
    /// Tasks to implement, ordered by priority
    pub tasks: Vec<ImplementationTask>,
    /// Estimated total development time (hours)
    pub estimated_hours: f64,
    /// Expected performance improvement
    pub expected_speedup: f64,
}

/// Single implementation task
#[derive(Debug, Clone)]
pub struct ImplementationTask {
    pub operation_name: String,
    pub priority: Priority,
    pub estimated_hours: f64,
    pub dependencies: Vec<String>,
    pub implementation_hints: String,
}

/// GPU gradient planner and analyzer
pub struct GpuGradientPlanner {
    operations: HashMap<String, GpuGradInfo>,
}

impl GpuGradientPlanner {
    /// Create a new GPU gradient planner
    pub fn new() -> Self {
        let mut planner = Self {
            operations: HashMap::new(),
        };
        planner.initialize_operations();
        planner
    }

    /// Analyze current GPU gradient coverage
    pub fn analyze_coverage(&self) -> GpuCoverageAnalysis {
        let total_count = self.operations.len();
        let gpu_count = self
            .operations
            .values()
            .filter(|op| matches!(op.status, GpuGradStatus::Implemented))
            .count();

        let coverage_percentage = if total_count > 0 {
            (gpu_count as f64 / total_count as f64) * 100.0
        } else {
            0.0
        };

        // Analyze by category
        let mut by_category: HashMap<String, GpuCategoryCoverage> = HashMap::new();
        for op in self.operations.values() {
            let entry = by_category
                .entry(op.category.clone())
                .or_insert(GpuCategoryCoverage {
                    total: 0,
                    gpu_supported: 0,
                    percentage: 0.0,
                });

            entry.total += 1;
            if matches!(op.status, GpuGradStatus::Implemented) {
                entry.gpu_supported += 1;
            }
        }

        // Calculate percentages
        for coverage in by_category.values_mut() {
            coverage.percentage = if coverage.total > 0 {
                (coverage.gpu_supported as f64 / coverage.total as f64) * 100.0
            } else {
                0.0
            };
        }

        // Group missing by priority
        let mut missing_by_priority: HashMap<Priority, Vec<String>> = HashMap::new();
        for op in self.operations.values() {
            if matches!(op.status, GpuGradStatus::Missing | GpuGradStatus::Planned) {
                missing_by_priority
                    .entry(op.priority)
                    .or_default()
                    .push(op.operation_name.clone());
            }
        }

        GpuCoverageAnalysis {
            total_count,
            gpu_count,
            coverage_percentage,
            by_category,
            missing_by_priority,
        }
    }

    /// Generate implementation plan for given priority threshold
    pub fn generate_implementation_plan(&self, min_priority: Priority) -> ImplementationPlan {
        let mut tasks = Vec::new();
        let mut total_hours = 0.0;
        let mut expected_speedup = 1.0;

        // Collect operations to implement
        let mut to_implement: Vec<_> = self
            .operations
            .values()
            .filter(|op| {
                matches!(op.status, GpuGradStatus::Missing | GpuGradStatus::Planned)
                    && op.priority >= min_priority
            })
            .collect();

        // Sort by priority (highest first) and then by estimated speedup
        to_implement.sort_by(|a, b| {
            b.priority.cmp(&a.priority).then_with(|| {
                b.estimated_speedup
                    .partial_cmp(&a.estimated_speedup)
                    .unwrap()
            })
        });

        for op in to_implement {
            let hours = Self::estimate_implementation_time(op);
            total_hours += hours;
            expected_speedup += op.estimated_speedup * 0.1; // Weighted contribution

            tasks.push(ImplementationTask {
                operation_name: op.operation_name.clone(),
                priority: op.priority,
                estimated_hours: hours,
                dependencies: op.dependencies.clone(),
                implementation_hints: op.notes.clone(),
            });
        }

        ImplementationPlan {
            tasks,
            estimated_hours: total_hours,
            expected_speedup,
        }
    }

    /// Get operations by category
    pub fn get_operations_by_category(&self, category: &str) -> Vec<&GpuGradInfo> {
        self.operations
            .values()
            .filter(|op| op.category == category)
            .collect()
    }

    /// Print detailed coverage report
    pub fn print_coverage_report(&self) {
        let analysis = self.analyze_coverage();

        println!("\n=== GPU Gradient Coverage Report ===");
        println!(
            "Overall Coverage: {}/{} ({:.1}%)",
            analysis.gpu_count, analysis.total_count, analysis.coverage_percentage
        );

        println!("\n--- Coverage by Category ---");
        let mut categories: Vec<_> = analysis.by_category.iter().collect();
        categories.sort_by_key(|(name, _)| (*name).to_string());

        for (category, coverage) in categories {
            println!(
                "{}: {}/{} ({:.1}%)",
                category, coverage.gpu_supported, coverage.total, coverage.percentage
            );
        }

        println!("\n--- Missing Operations by Priority ---");
        for priority in &[
            Priority::Critical,
            Priority::High,
            Priority::Medium,
            Priority::Low,
        ] {
            if let Some(ops) = analysis.missing_by_priority.get(priority) {
                println!("\n{:?} Priority ({} ops):", priority, ops.len());
                for op in ops {
                    println!("  - {}", op);
                }
            }
        }
        println!("\n====================================\n");
    }

    // Private helper methods

    fn initialize_operations(&mut self) {
        // Linear Algebra Operations
        self.add_operation(GpuGradInfo {
            operation_name: "matmul".to_string(),
            category: "LinearAlgebra".to_string(),
            status: GpuGradStatus::Implemented,
            priority: Priority::Critical,
            estimated_speedup: 15.0,
            memory_overhead_bytes: 8,
            complexity: 6,
            dependencies: vec![],
            notes: "Already optimized with cuBLAS/rocBLAS".to_string(),
        });

        self.add_operation(GpuGradInfo {
            operation_name: "conv2d".to_string(),
            category: "Convolution".to_string(),
            status: GpuGradStatus::Implemented,
            priority: Priority::Critical,
            estimated_speedup: 20.0,
            memory_overhead_bytes: 16,
            complexity: 8,
            dependencies: vec![],
            notes: "cuDNN integration for conv backward".to_string(),
        });

        // Activation Functions
        self.add_operation(GpuGradInfo {
            operation_name: "relu".to_string(),
            category: "Activations".to_string(),
            status: GpuGradStatus::Implemented,
            priority: Priority::High,
            estimated_speedup: 10.0,
            memory_overhead_bytes: 1,
            complexity: 2,
            dependencies: vec![],
            notes: "Simple element-wise kernel".to_string(),
        });

        self.add_operation(GpuGradInfo {
            operation_name: "gelu".to_string(),
            category: "Activations".to_string(),
            status: GpuGradStatus::Missing,
            priority: Priority::High,
            estimated_speedup: 8.0,
            memory_overhead_bytes: 4,
            complexity: 3,
            dependencies: vec![],
            notes: "Implement GELU gradient: 0.5 * (1 + tanh(...)) + correction term".to_string(),
        });

        self.add_operation(GpuGradInfo {
            operation_name: "silu".to_string(),
            category: "Activations".to_string(),
            status: GpuGradStatus::Missing,
            priority: Priority::High,
            estimated_speedup: 7.0,
            memory_overhead_bytes: 4,
            complexity: 3,
            dependencies: vec!["sigmoid".to_string()],
            notes: "SiLU grad: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))".to_string(),
        });

        // Normalization
        self.add_operation(GpuGradInfo {
            operation_name: "batch_norm".to_string(),
            category: "Normalization".to_string(),
            status: GpuGradStatus::Partial,
            priority: Priority::Critical,
            estimated_speedup: 12.0,
            memory_overhead_bytes: 24,
            complexity: 7,
            dependencies: vec![],
            notes: "Need to implement running stats update on GPU".to_string(),
        });

        self.add_operation(GpuGradInfo {
            operation_name: "layer_norm".to_string(),
            category: "Normalization".to_string(),
            status: GpuGradStatus::Missing,
            priority: Priority::Critical,
            estimated_speedup: 10.0,
            memory_overhead_bytes: 16,
            complexity: 6,
            dependencies: vec![],
            notes: "Critical for transformer models. Fuse mean/variance computation".to_string(),
        });

        self.add_operation(GpuGradInfo {
            operation_name: "group_norm".to_string(),
            category: "Normalization".to_string(),
            status: GpuGradStatus::Missing,
            priority: Priority::High,
            estimated_speedup: 9.0,
            memory_overhead_bytes: 16,
            complexity: 7,
            dependencies: vec![],
            notes: "Similar to layer_norm but with group dimension".to_string(),
        });

        // Reductions
        self.add_operation(GpuGradInfo {
            operation_name: "sum".to_string(),
            category: "Reductions".to_string(),
            status: GpuGradStatus::Implemented,
            priority: Priority::Critical,
            estimated_speedup: 12.0,
            memory_overhead_bytes: 4,
            complexity: 5,
            dependencies: vec![],
            notes: "Tree reduction pattern".to_string(),
        });

        self.add_operation(GpuGradInfo {
            operation_name: "mean".to_string(),
            category: "Reductions".to_string(),
            status: GpuGradStatus::Implemented,
            priority: Priority::High,
            estimated_speedup: 11.0,
            memory_overhead_bytes: 4,
            complexity: 5,
            dependencies: vec![],
            notes: "Based on sum + scale".to_string(),
        });

        self.add_operation(GpuGradInfo {
            operation_name: "softmax".to_string(),
            category: "Activations".to_string(),
            status: GpuGradStatus::Partial,
            priority: Priority::Critical,
            estimated_speedup: 14.0,
            memory_overhead_bytes: 8,
            complexity: 6,
            dependencies: vec!["max".to_string(), "sum".to_string()],
            notes: "Numerically stable version with max subtraction. Optimize for last dim"
                .to_string(),
        });

        // Attention operations
        self.add_operation(GpuGradInfo {
            operation_name: "scaled_dot_product_attention".to_string(),
            category: "Attention".to_string(),
            status: GpuGradStatus::Missing,
            priority: Priority::Critical,
            estimated_speedup: 25.0,
            memory_overhead_bytes: 32,
            complexity: 9,
            dependencies: vec!["matmul".to_string(), "softmax".to_string()],
            notes: "Fused attention kernel critical for transformers. Flash Attention style"
                .to_string(),
        });

        // Loss functions
        self.add_operation(GpuGradInfo {
            operation_name: "cross_entropy".to_string(),
            category: "Loss".to_string(),
            status: GpuGradStatus::Missing,
            priority: Priority::Critical,
            estimated_speedup: 10.0,
            memory_overhead_bytes: 8,
            complexity: 5,
            dependencies: vec!["log_softmax".to_string()],
            notes: "Fuse log_softmax + nll_loss for numerical stability".to_string(),
        });

        self.add_operation(GpuGradInfo {
            operation_name: "mse_loss".to_string(),
            category: "Loss".to_string(),
            status: GpuGradStatus::Missing,
            priority: Priority::High,
            estimated_speedup: 8.0,
            memory_overhead_bytes: 4,
            complexity: 3,
            dependencies: vec![],
            notes: "Simple: 2 * (pred - target) / n".to_string(),
        });

        // Advanced operations
        self.add_operation(GpuGradInfo {
            operation_name: "einsum".to_string(),
            category: "Advanced".to_string(),
            status: GpuGradStatus::Missing,
            priority: Priority::Medium,
            estimated_speedup: 15.0,
            memory_overhead_bytes: 16,
            complexity: 9,
            dependencies: vec![],
            notes: "Complex einsum backward. May require multiple kernels".to_string(),
        });

        self.add_operation(GpuGradInfo {
            operation_name: "gather".to_string(),
            category: "Indexing".to_string(),
            status: GpuGradStatus::Missing,
            priority: Priority::High,
            estimated_speedup: 12.0,
            memory_overhead_bytes: 8,
            complexity: 6,
            dependencies: vec![],
            notes: "Scatter gradient back to source indices".to_string(),
        });

        self.add_operation(GpuGradInfo {
            operation_name: "scatter".to_string(),
            category: "Indexing".to_string(),
            status: GpuGradStatus::Missing,
            priority: Priority::High,
            estimated_speedup: 12.0,
            memory_overhead_bytes: 8,
            complexity: 6,
            dependencies: vec![],
            notes: "Gather gradient from scattered indices".to_string(),
        });
    }

    fn add_operation(&mut self, info: GpuGradInfo) {
        self.operations.insert(info.operation_name.clone(), info);
    }

    fn estimate_implementation_time(op: &GpuGradInfo) -> f64 {
        // Base time + complexity factor
        let base_hours = match op.complexity {
            1..=3 => 4.0,
            4..=6 => 8.0,
            7..=9 => 16.0,
            _ => 24.0,
        };

        // Add time for dependencies
        let dependency_hours = op.dependencies.len() as f64 * 2.0;

        base_hours + dependency_hours
    }
}

impl Default for GpuGradientPlanner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_planner_creation() {
        let planner = GpuGradientPlanner::new();
        assert!(!planner.operations.is_empty());
    }

    #[test]
    fn test_coverage_analysis() {
        let planner = GpuGradientPlanner::new();
        let analysis = planner.analyze_coverage();

        assert!(analysis.total_count > 0);
        assert!(analysis.coverage_percentage <= 100.0);
        assert!(!analysis.by_category.is_empty());
    }

    #[test]
    fn test_implementation_plan() {
        let planner = GpuGradientPlanner::new();
        let plan = planner.generate_implementation_plan(Priority::High);

        assert!(!plan.tasks.is_empty());
        assert!(plan.estimated_hours > 0.0);

        // Check tasks are sorted by priority
        for window in plan.tasks.windows(2) {
            assert!(window[0].priority >= window[1].priority);
        }
    }

    #[test]
    fn test_get_operations_by_category() {
        let planner = GpuGradientPlanner::new();
        let activations = planner.get_operations_by_category("Activations");

        assert!(!activations.is_empty());
        for op in activations {
            assert_eq!(op.category, "Activations");
        }
    }

    #[test]
    fn test_priority_ordering() {
        assert!(Priority::Critical > Priority::High);
        assert!(Priority::High > Priority::Medium);
        assert!(Priority::Medium > Priority::Low);
    }
}
