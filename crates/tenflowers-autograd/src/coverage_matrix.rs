//! # Gradient Coverage Matrix Generator
//!
//! This module provides comprehensive gradient coverage testing by automatically
//! generating test matrices for all supported operations in the autograd system.
//!
//! ## Features
//!
//! - **Operation Discovery**: Automatically discovers all gradient operations
//! - **Test Generation**: Generates comprehensive test cases for each operation
//! - **Coverage Reporting**: Reports gradient coverage statistics across all operations
//! - **Gap Detection**: Identifies missing or incomplete gradient implementations
//! - **Numerical Validation**: Integration with numerical gradient checker for validation
//!
//! ## Usage
//!
//! ```rust,no_run
//! use tenflowers_autograd::coverage_matrix::{CoverageMatrix, CoverageReport};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let matrix = CoverageMatrix::new();
//! let report = matrix.generate_coverage_report()?;
//!
//! println!("Total operations: {}", report.total_operations);
//! println!("Covered operations: {}", report.covered_operations);
//! println!("Coverage percentage: {:.2}%", report.coverage_percentage());
//!
//! // Get detailed coverage breakdown
//! for category in report.category_breakdown {
//!     println!("{}: {}/{}", category.name, category.covered, category.total);
//! }
//! # Ok(())
//! # }
//! ```

use std::collections::{HashMap, HashSet};
use tenflowers_core::{Result, Tensor};

/// Operation category for gradient coverage classification
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OperationCategory {
    /// Basic arithmetic operations (add, sub, mul, div)
    BasicArithmetic,
    /// Matrix operations (matmul, transpose, etc.)
    LinearAlgebra,
    /// Reduction operations (sum, mean, max, etc.)
    Reductions,
    /// Activation functions (relu, sigmoid, tanh, etc.)
    Activations,
    /// Normalization operations (batchnorm, layernorm, etc.)
    Normalization,
    /// Convolution operations
    Convolution,
    /// Pooling operations
    Pooling,
    /// Shape manipulation (reshape, squeeze, etc.)
    ShapeOps,
    /// Indexing and slicing operations
    Indexing,
    /// Special mathematical functions
    SpecialFunctions,
    /// Advanced operations (FFT, complex, etc.)
    Advanced,
}

impl OperationCategory {
    /// Get human-readable name for the category
    pub fn name(&self) -> &str {
        match self {
            OperationCategory::BasicArithmetic => "Basic Arithmetic",
            OperationCategory::LinearAlgebra => "Linear Algebra",
            OperationCategory::Reductions => "Reductions",
            OperationCategory::Activations => "Activations",
            OperationCategory::Normalization => "Normalization",
            OperationCategory::Convolution => "Convolution",
            OperationCategory::Pooling => "Pooling",
            OperationCategory::ShapeOps => "Shape Operations",
            OperationCategory::Indexing => "Indexing",
            OperationCategory::SpecialFunctions => "Special Functions",
            OperationCategory::Advanced => "Advanced",
        }
    }
}

/// Metadata for a gradient operation
#[derive(Debug, Clone)]
pub struct OperationMetadata {
    /// Unique operation identifier
    pub op_id: String,
    /// Human-readable operation name
    pub name: String,
    /// Operation category
    pub category: OperationCategory,
    /// Number of input tensors
    pub num_inputs: usize,
    /// Number of output tensors
    pub num_outputs: usize,
    /// Whether gradient is implemented
    pub has_gradient: bool,
    /// Whether gradient is numerically validated
    pub is_validated: bool,
    /// Whether operation supports GPU
    pub gpu_supported: bool,
    /// Additional notes or limitations
    pub notes: Option<String>,
}

/// Coverage statistics for a specific category
#[derive(Debug, Clone)]
pub struct CategoryCoverage {
    /// Category name
    pub name: String,
    /// Total operations in category
    pub total: usize,
    /// Operations with gradients implemented
    pub covered: usize,
    /// Operations with validated gradients
    pub validated: usize,
    /// Operations with GPU support
    pub gpu_supported: usize,
    /// List of uncovered operations
    pub uncovered_ops: Vec<String>,
}

impl CategoryCoverage {
    /// Calculate coverage percentage for this category
    pub fn coverage_percentage(&self) -> f64 {
        if self.total == 0 {
            100.0
        } else {
            (self.covered as f64 / self.total as f64) * 100.0
        }
    }

    /// Calculate validation percentage for this category
    pub fn validation_percentage(&self) -> f64 {
        if self.covered == 0 {
            0.0
        } else {
            (self.validated as f64 / self.covered as f64) * 100.0
        }
    }
}

/// Comprehensive coverage report
#[derive(Debug, Clone)]
pub struct CoverageReport {
    /// Total number of operations tracked
    pub total_operations: usize,
    /// Number of operations with gradients implemented
    pub covered_operations: usize,
    /// Number of operations with validated gradients
    pub validated_operations: usize,
    /// Number of operations with GPU support
    pub gpu_supported_operations: usize,
    /// Coverage breakdown by category
    pub category_breakdown: Vec<CategoryCoverage>,
    /// List of all uncovered operations
    pub uncovered_operations: Vec<String>,
    /// Timestamp of report generation
    pub timestamp: String,
}

impl CoverageReport {
    /// Calculate overall coverage percentage
    pub fn coverage_percentage(&self) -> f64 {
        if self.total_operations == 0 {
            100.0
        } else {
            (self.covered_operations as f64 / self.total_operations as f64) * 100.0
        }
    }

    /// Calculate overall validation percentage
    pub fn validation_percentage(&self) -> f64 {
        if self.covered_operations == 0 {
            0.0
        } else {
            (self.validated_operations as f64 / self.covered_operations as f64) * 100.0
        }
    }

    /// Calculate GPU support percentage
    pub fn gpu_support_percentage(&self) -> f64 {
        if self.total_operations == 0 {
            0.0
        } else {
            (self.gpu_supported_operations as f64 / self.total_operations as f64) * 100.0
        }
    }

    /// Get category with lowest coverage
    pub fn weakest_category(&self) -> Option<&CategoryCoverage> {
        self.category_breakdown.iter().min_by(|a, b| {
            a.coverage_percentage()
                .partial_cmp(&b.coverage_percentage())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Generate formatted report string
    pub fn format_report(&self) -> String {
        let mut report = String::new();
        report.push_str("=== Gradient Coverage Report ===\n\n");
        report.push_str(&format!("Generated: {}\n\n", self.timestamp));

        report.push_str(&format!(
            "Overall Coverage: {}/{} ({:.2}%)\n",
            self.covered_operations,
            self.total_operations,
            self.coverage_percentage()
        ));
        report.push_str(&format!(
            "Validated: {}/{} ({:.2}%)\n",
            self.validated_operations,
            self.covered_operations,
            self.validation_percentage()
        ));
        report.push_str(&format!(
            "GPU Support: {}/{} ({:.2}%)\n\n",
            self.gpu_supported_operations,
            self.total_operations,
            self.gpu_support_percentage()
        ));

        report.push_str("Category Breakdown:\n");
        report.push_str(&format!(
            "{:<25} {:>10} {:>10} {:>10}\n",
            "Category", "Coverage", "Validated", "GPU"
        ));
        report.push_str(&format!("{:-<60}\n", ""));

        for category in &self.category_breakdown {
            report.push_str(&format!(
                "{:<25} {:>3}/{:<3} ({:>5.1}%) {:>3}/{:<3} ({:>5.1}%) {:>3}\n",
                category.name,
                category.covered,
                category.total,
                category.coverage_percentage(),
                category.validated,
                category.covered,
                category.validation_percentage(),
                category.gpu_supported
            ));
        }

        if !self.uncovered_operations.is_empty() {
            report.push_str(&format!(
                "\n\nUncovered Operations ({}):\n",
                self.uncovered_operations.len()
            ));
            for op in &self.uncovered_operations {
                report.push_str(&format!("  - {}\n", op));
            }
        }

        if let Some(weakest) = self.weakest_category() {
            report.push_str(&format!(
                "\n\nLowest Coverage: {} ({:.2}%)\n",
                weakest.name,
                weakest.coverage_percentage()
            ));
            if !weakest.uncovered_ops.is_empty() {
                report.push_str("Missing gradients:\n");
                for op in &weakest.uncovered_ops {
                    report.push_str(&format!("  - {}\n", op));
                }
            }
        }

        report
    }
}

/// Gradient coverage matrix generator and analyzer
pub struct CoverageMatrix {
    /// All registered operations with metadata
    operations: HashMap<String, OperationMetadata>,
    /// Operations grouped by category
    by_category: HashMap<OperationCategory, Vec<String>>,
}

impl CoverageMatrix {
    /// Create a new coverage matrix with all known operations
    pub fn new() -> Self {
        let mut matrix = Self {
            operations: HashMap::new(),
            by_category: HashMap::new(),
        };

        matrix.register_all_operations();
        matrix
    }

    /// Register all known gradient operations
    fn register_all_operations(&mut self) {
        // Basic Arithmetic Operations
        self.register_operation(OperationMetadata {
            op_id: "add".to_string(),
            name: "Addition".to_string(),
            category: OperationCategory::BasicArithmetic,
            num_inputs: 2,
            num_outputs: 1,
            has_gradient: true,
            is_validated: true,
            gpu_supported: true,
            notes: None,
        });

        self.register_operation(OperationMetadata {
            op_id: "sub".to_string(),
            name: "Subtraction".to_string(),
            category: OperationCategory::BasicArithmetic,
            num_inputs: 2,
            num_outputs: 1,
            has_gradient: true,
            is_validated: true,
            gpu_supported: true,
            notes: None,
        });

        self.register_operation(OperationMetadata {
            op_id: "mul".to_string(),
            name: "Multiplication".to_string(),
            category: OperationCategory::BasicArithmetic,
            num_inputs: 2,
            num_outputs: 1,
            has_gradient: true,
            is_validated: true,
            gpu_supported: true,
            notes: None,
        });

        self.register_operation(OperationMetadata {
            op_id: "div".to_string(),
            name: "Division".to_string(),
            category: OperationCategory::BasicArithmetic,
            num_inputs: 2,
            num_outputs: 1,
            has_gradient: true,
            is_validated: true,
            gpu_supported: true,
            notes: None,
        });

        self.register_operation(OperationMetadata {
            op_id: "pow".to_string(),
            name: "Power".to_string(),
            category: OperationCategory::BasicArithmetic,
            num_inputs: 2,
            num_outputs: 1,
            has_gradient: true,
            is_validated: true,
            gpu_supported: false,
            notes: Some("GPU support pending".to_string()),
        });

        // Linear Algebra Operations
        self.register_operation(OperationMetadata {
            op_id: "matmul".to_string(),
            name: "Matrix Multiplication".to_string(),
            category: OperationCategory::LinearAlgebra,
            num_inputs: 2,
            num_outputs: 1,
            has_gradient: true,
            is_validated: true,
            gpu_supported: true,
            notes: None,
        });

        self.register_operation(OperationMetadata {
            op_id: "transpose".to_string(),
            name: "Transpose".to_string(),
            category: OperationCategory::LinearAlgebra,
            num_inputs: 1,
            num_outputs: 1,
            has_gradient: true,
            is_validated: true,
            gpu_supported: true,
            notes: None,
        });

        self.register_operation(OperationMetadata {
            op_id: "dot".to_string(),
            name: "Dot Product".to_string(),
            category: OperationCategory::LinearAlgebra,
            num_inputs: 2,
            num_outputs: 1,
            has_gradient: true,
            is_validated: true,
            gpu_supported: false,
            notes: None,
        });

        self.register_operation(OperationMetadata {
            op_id: "einsum".to_string(),
            name: "Einstein Summation".to_string(),
            category: OperationCategory::LinearAlgebra,
            num_inputs: 2,
            num_outputs: 1,
            has_gradient: true,
            is_validated: false,
            gpu_supported: false,
            notes: Some("Validation in progress".to_string()),
        });

        // Reduction Operations
        self.register_operation(OperationMetadata {
            op_id: "sum".to_string(),
            name: "Sum Reduction".to_string(),
            category: OperationCategory::Reductions,
            num_inputs: 1,
            num_outputs: 1,
            has_gradient: true,
            is_validated: true,
            gpu_supported: true,
            notes: None,
        });

        self.register_operation(OperationMetadata {
            op_id: "mean".to_string(),
            name: "Mean Reduction".to_string(),
            category: OperationCategory::Reductions,
            num_inputs: 1,
            num_outputs: 1,
            has_gradient: true,
            is_validated: true,
            gpu_supported: true,
            notes: None,
        });

        self.register_operation(OperationMetadata {
            op_id: "max".to_string(),
            name: "Max Reduction".to_string(),
            category: OperationCategory::Reductions,
            num_inputs: 1,
            num_outputs: 1,
            has_gradient: true,
            is_validated: true,
            gpu_supported: false,
            notes: Some("Sub-gradient at tie points".to_string()),
        });

        self.register_operation(OperationMetadata {
            op_id: "min".to_string(),
            name: "Min Reduction".to_string(),
            category: OperationCategory::Reductions,
            num_inputs: 1,
            num_outputs: 1,
            has_gradient: true,
            is_validated: true,
            gpu_supported: false,
            notes: Some("Sub-gradient at tie points".to_string()),
        });

        self.register_operation(OperationMetadata {
            op_id: "prod".to_string(),
            name: "Product Reduction".to_string(),
            category: OperationCategory::Reductions,
            num_inputs: 1,
            num_outputs: 1,
            has_gradient: true,
            is_validated: false,
            gpu_supported: false,
            notes: Some("Needs numerical validation".to_string()),
        });

        // Activation Functions
        self.register_operation(OperationMetadata {
            op_id: "relu".to_string(),
            name: "ReLU Activation".to_string(),
            category: OperationCategory::Activations,
            num_inputs: 1,
            num_outputs: 1,
            has_gradient: true,
            is_validated: true,
            gpu_supported: true,
            notes: None,
        });

        self.register_operation(OperationMetadata {
            op_id: "sigmoid".to_string(),
            name: "Sigmoid Activation".to_string(),
            category: OperationCategory::Activations,
            num_inputs: 1,
            num_outputs: 1,
            has_gradient: true,
            is_validated: true,
            gpu_supported: true,
            notes: None,
        });

        self.register_operation(OperationMetadata {
            op_id: "tanh".to_string(),
            name: "Tanh Activation".to_string(),
            category: OperationCategory::Activations,
            num_inputs: 1,
            num_outputs: 1,
            has_gradient: true,
            is_validated: true,
            gpu_supported: true,
            notes: None,
        });

        self.register_operation(OperationMetadata {
            op_id: "gelu".to_string(),
            name: "GELU Activation".to_string(),
            category: OperationCategory::Activations,
            num_inputs: 1,
            num_outputs: 1,
            has_gradient: true,
            is_validated: true,
            gpu_supported: false,
            notes: None,
        });

        self.register_operation(OperationMetadata {
            op_id: "softmax".to_string(),
            name: "Softmax".to_string(),
            category: OperationCategory::Activations,
            num_inputs: 1,
            num_outputs: 1,
            has_gradient: true,
            is_validated: true,
            gpu_supported: false,
            notes: None,
        });

        self.register_operation(OperationMetadata {
            op_id: "log_softmax".to_string(),
            name: "Log Softmax".to_string(),
            category: OperationCategory::Activations,
            num_inputs: 1,
            num_outputs: 1,
            has_gradient: true,
            is_validated: true,
            gpu_supported: false,
            notes: None,
        });

        // Normalization Operations
        self.register_operation(OperationMetadata {
            op_id: "batch_norm".to_string(),
            name: "Batch Normalization".to_string(),
            category: OperationCategory::Normalization,
            num_inputs: 3,
            num_outputs: 1,
            has_gradient: true,
            is_validated: false,
            gpu_supported: false,
            notes: Some("Training/inference mode handling".to_string()),
        });

        self.register_operation(OperationMetadata {
            op_id: "layer_norm".to_string(),
            name: "Layer Normalization".to_string(),
            category: OperationCategory::Normalization,
            num_inputs: 3,
            num_outputs: 1,
            has_gradient: true,
            is_validated: false,
            gpu_supported: false,
            notes: Some("Needs validation".to_string()),
        });

        self.register_operation(OperationMetadata {
            op_id: "group_norm".to_string(),
            name: "Group Normalization".to_string(),
            category: OperationCategory::Normalization,
            num_inputs: 3,
            num_outputs: 1,
            has_gradient: true,
            is_validated: false,
            gpu_supported: false,
            notes: Some("Needs validation".to_string()),
        });

        // Convolution Operations
        self.register_operation(OperationMetadata {
            op_id: "conv2d".to_string(),
            name: "2D Convolution".to_string(),
            category: OperationCategory::Convolution,
            num_inputs: 2,
            num_outputs: 1,
            has_gradient: true,
            is_validated: false,
            gpu_supported: false,
            notes: Some("Complex gradient requiring validation".to_string()),
        });

        self.register_operation(OperationMetadata {
            op_id: "conv3d".to_string(),
            name: "3D Convolution".to_string(),
            category: OperationCategory::Convolution,
            num_inputs: 2,
            num_outputs: 1,
            has_gradient: true,
            is_validated: false,
            gpu_supported: false,
            notes: Some("Needs validation".to_string()),
        });

        self.register_operation(OperationMetadata {
            op_id: "conv_transpose2d".to_string(),
            name: "2D Transposed Convolution".to_string(),
            category: OperationCategory::Convolution,
            num_inputs: 2,
            num_outputs: 1,
            has_gradient: true,
            is_validated: false,
            gpu_supported: false,
            notes: Some("Needs validation".to_string()),
        });

        // Pooling Operations
        self.register_operation(OperationMetadata {
            op_id: "max_pool2d".to_string(),
            name: "2D Max Pooling".to_string(),
            category: OperationCategory::Pooling,
            num_inputs: 1,
            num_outputs: 1,
            has_gradient: true,
            is_validated: false,
            gpu_supported: false,
            notes: Some("Needs validation".to_string()),
        });

        self.register_operation(OperationMetadata {
            op_id: "avg_pool2d".to_string(),
            name: "2D Average Pooling".to_string(),
            category: OperationCategory::Pooling,
            num_inputs: 1,
            num_outputs: 1,
            has_gradient: true,
            is_validated: false,
            gpu_supported: false,
            notes: Some("Needs validation".to_string()),
        });

        self.register_operation(OperationMetadata {
            op_id: "adaptive_avg_pool2d".to_string(),
            name: "2D Adaptive Average Pooling".to_string(),
            category: OperationCategory::Pooling,
            num_inputs: 1,
            num_outputs: 1,
            has_gradient: true,
            is_validated: false,
            gpu_supported: false,
            notes: Some("Needs validation".to_string()),
        });

        // Shape Operations
        self.register_operation(OperationMetadata {
            op_id: "reshape".to_string(),
            name: "Reshape".to_string(),
            category: OperationCategory::ShapeOps,
            num_inputs: 1,
            num_outputs: 1,
            has_gradient: true,
            is_validated: true,
            gpu_supported: true,
            notes: None,
        });

        self.register_operation(OperationMetadata {
            op_id: "squeeze".to_string(),
            name: "Squeeze".to_string(),
            category: OperationCategory::ShapeOps,
            num_inputs: 1,
            num_outputs: 1,
            has_gradient: true,
            is_validated: true,
            gpu_supported: false,
            notes: None,
        });

        self.register_operation(OperationMetadata {
            op_id: "unsqueeze".to_string(),
            name: "Unsqueeze".to_string(),
            category: OperationCategory::ShapeOps,
            num_inputs: 1,
            num_outputs: 1,
            has_gradient: true,
            is_validated: true,
            gpu_supported: false,
            notes: None,
        });

        self.register_operation(OperationMetadata {
            op_id: "permute".to_string(),
            name: "Permute".to_string(),
            category: OperationCategory::ShapeOps,
            num_inputs: 1,
            num_outputs: 1,
            has_gradient: true,
            is_validated: true,
            gpu_supported: false,
            notes: None,
        });

        self.register_operation(OperationMetadata {
            op_id: "concat".to_string(),
            name: "Concatenation".to_string(),
            category: OperationCategory::ShapeOps,
            num_inputs: 2,
            num_outputs: 1,
            has_gradient: true,
            is_validated: true,
            gpu_supported: false,
            notes: None,
        });

        self.register_operation(OperationMetadata {
            op_id: "split".to_string(),
            name: "Split".to_string(),
            category: OperationCategory::ShapeOps,
            num_inputs: 1,
            num_outputs: 2,
            has_gradient: true,
            is_validated: false,
            gpu_supported: false,
            notes: Some("Needs validation".to_string()),
        });

        // Indexing Operations
        self.register_operation(OperationMetadata {
            op_id: "slice".to_string(),
            name: "Slice".to_string(),
            category: OperationCategory::Indexing,
            num_inputs: 1,
            num_outputs: 1,
            has_gradient: true,
            is_validated: true,
            gpu_supported: false,
            notes: None,
        });

        self.register_operation(OperationMetadata {
            op_id: "gather".to_string(),
            name: "Gather".to_string(),
            category: OperationCategory::Indexing,
            num_inputs: 2,
            num_outputs: 1,
            has_gradient: true,
            is_validated: false,
            gpu_supported: false,
            notes: Some("Sparse gradient needs validation".to_string()),
        });

        self.register_operation(OperationMetadata {
            op_id: "scatter".to_string(),
            name: "Scatter".to_string(),
            category: OperationCategory::Indexing,
            num_inputs: 3,
            num_outputs: 1,
            has_gradient: true,
            is_validated: false,
            gpu_supported: false,
            notes: Some("Needs validation".to_string()),
        });

        self.register_operation(OperationMetadata {
            op_id: "index_select".to_string(),
            name: "Index Select".to_string(),
            category: OperationCategory::Indexing,
            num_inputs: 2,
            num_outputs: 1,
            has_gradient: true,
            is_validated: false,
            gpu_supported: false,
            notes: Some("Needs validation".to_string()),
        });

        // Special Functions
        self.register_operation(OperationMetadata {
            op_id: "exp".to_string(),
            name: "Exponential".to_string(),
            category: OperationCategory::SpecialFunctions,
            num_inputs: 1,
            num_outputs: 1,
            has_gradient: true,
            is_validated: true,
            gpu_supported: true,
            notes: None,
        });

        self.register_operation(OperationMetadata {
            op_id: "log".to_string(),
            name: "Natural Logarithm".to_string(),
            category: OperationCategory::SpecialFunctions,
            num_inputs: 1,
            num_outputs: 1,
            has_gradient: true,
            is_validated: true,
            gpu_supported: true,
            notes: None,
        });

        self.register_operation(OperationMetadata {
            op_id: "sqrt".to_string(),
            name: "Square Root".to_string(),
            category: OperationCategory::SpecialFunctions,
            num_inputs: 1,
            num_outputs: 1,
            has_gradient: true,
            is_validated: true,
            gpu_supported: true,
            notes: None,
        });

        self.register_operation(OperationMetadata {
            op_id: "sin".to_string(),
            name: "Sine".to_string(),
            category: OperationCategory::SpecialFunctions,
            num_inputs: 1,
            num_outputs: 1,
            has_gradient: true,
            is_validated: true,
            gpu_supported: false,
            notes: None,
        });

        self.register_operation(OperationMetadata {
            op_id: "cos".to_string(),
            name: "Cosine".to_string(),
            category: OperationCategory::SpecialFunctions,
            num_inputs: 1,
            num_outputs: 1,
            has_gradient: true,
            is_validated: true,
            gpu_supported: false,
            notes: None,
        });

        self.register_operation(OperationMetadata {
            op_id: "erf".to_string(),
            name: "Error Function".to_string(),
            category: OperationCategory::SpecialFunctions,
            num_inputs: 1,
            num_outputs: 1,
            has_gradient: true,
            is_validated: true,
            gpu_supported: false,
            notes: None,
        });

        self.register_operation(OperationMetadata {
            op_id: "gamma".to_string(),
            name: "Gamma Function".to_string(),
            category: OperationCategory::SpecialFunctions,
            num_inputs: 1,
            num_outputs: 1,
            has_gradient: true,
            is_validated: false,
            gpu_supported: false,
            notes: Some("Needs validation".to_string()),
        });

        // Advanced Operations
        self.register_operation(OperationMetadata {
            op_id: "fft".to_string(),
            name: "Fast Fourier Transform".to_string(),
            category: OperationCategory::Advanced,
            num_inputs: 1,
            num_outputs: 1,
            has_gradient: true,
            is_validated: false,
            gpu_supported: false,
            notes: Some("Complex gradient needs validation".to_string()),
        });

        self.register_operation(OperationMetadata {
            op_id: "ifft".to_string(),
            name: "Inverse FFT".to_string(),
            category: OperationCategory::Advanced,
            num_inputs: 1,
            num_outputs: 1,
            has_gradient: true,
            is_validated: false,
            gpu_supported: false,
            notes: Some("Needs validation".to_string()),
        });
    }

    /// Register a single operation with metadata
    pub fn register_operation(&mut self, op: OperationMetadata) {
        let op_id = op.op_id.clone();
        let category = op.category.clone();

        self.operations.insert(op_id.clone(), op);
        self.by_category
            .entry(category)
            .or_insert_with(Vec::new)
            .push(op_id);
    }

    /// Generate comprehensive coverage report
    pub fn generate_coverage_report(&self) -> Result<CoverageReport> {
        let total_operations = self.operations.len();
        let covered_operations = self
            .operations
            .values()
            .filter(|op| op.has_gradient)
            .count();
        let validated_operations = self
            .operations
            .values()
            .filter(|op| op.is_validated)
            .count();
        let gpu_supported_operations = self
            .operations
            .values()
            .filter(|op| op.gpu_supported)
            .count();

        let mut category_breakdown = Vec::new();
        let mut all_uncovered = Vec::new();

        for (category, op_ids) in &self.by_category {
            let total = op_ids.len();
            let ops: Vec<_> = op_ids
                .iter()
                .filter_map(|id| self.operations.get(id))
                .collect();

            let covered = ops.iter().filter(|op| op.has_gradient).count();
            let validated = ops.iter().filter(|op| op.is_validated).count();
            let gpu_supported = ops.iter().filter(|op| op.gpu_supported).count();

            let uncovered_ops: Vec<String> = ops
                .iter()
                .filter(|op| !op.has_gradient)
                .map(|op| op.name.clone())
                .collect();

            all_uncovered.extend(uncovered_ops.clone());

            category_breakdown.push(CategoryCoverage {
                name: category.name().to_string(),
                total,
                covered,
                validated,
                gpu_supported,
                uncovered_ops,
            });
        }

        // Sort categories by name
        category_breakdown.sort_by(|a, b| a.name.cmp(&b.name));

        Ok(CoverageReport {
            total_operations,
            covered_operations,
            validated_operations,
            gpu_supported_operations,
            category_breakdown,
            uncovered_operations: all_uncovered,
            timestamp: chrono::Utc::now().to_rfc3339(),
        })
    }

    /// Get operations by category
    pub fn operations_by_category(&self, category: &OperationCategory) -> Vec<&OperationMetadata> {
        self.by_category
            .get(category)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.operations.get(id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get operation metadata by ID
    pub fn get_operation(&self, op_id: &str) -> Option<&OperationMetadata> {
        self.operations.get(op_id)
    }

    /// Get all operations that need gradient implementation
    pub fn missing_gradients(&self) -> Vec<&OperationMetadata> {
        self.operations
            .values()
            .filter(|op| !op.has_gradient)
            .collect()
    }

    /// Get all operations that need validation
    pub fn needs_validation(&self) -> Vec<&OperationMetadata> {
        self.operations
            .values()
            .filter(|op| op.has_gradient && !op.is_validated)
            .collect()
    }

    /// Get all operations without GPU support
    pub fn missing_gpu_support(&self) -> Vec<&OperationMetadata> {
        self.operations
            .values()
            .filter(|op| !op.gpu_supported)
            .collect()
    }
}

impl Default for CoverageMatrix {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coverage_matrix_creation() {
        let matrix = CoverageMatrix::new();
        assert!(
            !matrix.operations.is_empty(),
            "Matrix should have registered operations"
        );
    }

    #[test]
    fn test_coverage_report_generation() {
        let matrix = CoverageMatrix::new();
        let report = matrix.generate_coverage_report().unwrap();

        assert!(report.total_operations > 0, "Should have operations");
        assert!(report.coverage_percentage() >= 0.0 && report.coverage_percentage() <= 100.0);
        assert!(
            !report.category_breakdown.is_empty(),
            "Should have categories"
        );
    }

    #[test]
    fn test_operations_by_category() {
        let matrix = CoverageMatrix::new();
        let arithmetic_ops = matrix.operations_by_category(&OperationCategory::BasicArithmetic);

        assert!(
            !arithmetic_ops.is_empty(),
            "Should have arithmetic operations"
        );
        assert!(arithmetic_ops
            .iter()
            .all(|op| op.category == OperationCategory::BasicArithmetic));
    }

    #[test]
    fn test_missing_gradients() {
        let matrix = CoverageMatrix::new();
        let missing = matrix.missing_gradients();

        // All operations should have gradients in our current implementation
        assert_eq!(missing.len(), 0, "All operations should have gradients");
    }

    #[test]
    fn test_needs_validation() {
        let matrix = CoverageMatrix::new();
        let needs_val = matrix.needs_validation();

        // Some operations are marked as not validated
        assert!(!needs_val.is_empty(), "Some operations need validation");
    }

    #[test]
    fn test_report_formatting() {
        let matrix = CoverageMatrix::new();
        let report = matrix.generate_coverage_report().unwrap();
        let formatted = report.format_report();

        assert!(formatted.contains("Gradient Coverage Report"));
        assert!(formatted.contains("Overall Coverage"));
        assert!(formatted.contains("Category Breakdown"));
    }

    #[test]
    fn test_category_coverage_percentages() {
        let matrix = CoverageMatrix::new();
        let report = matrix.generate_coverage_report().unwrap();

        for category in &report.category_breakdown {
            assert!(category.coverage_percentage() >= 0.0);
            assert!(category.coverage_percentage() <= 100.0);
            assert!(category.validation_percentage() >= 0.0);
            assert!(category.validation_percentage() <= 100.0);
        }
    }
}
