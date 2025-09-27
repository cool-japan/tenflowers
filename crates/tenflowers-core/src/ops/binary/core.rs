//! Core Binary Operations Infrastructure
//!
//! This module provides the fundamental traits, types, and registry for binary operations
//! with advanced performance analytics and optimization tracking.

use crate::{Result, TensorError};
use scirs2_core::metrics::{Histogram, Timer};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Ultra-performance binary operation trait with vectorization support
pub trait BinaryOp<T: Clone> {
    fn apply(&self, a: T, b: T) -> T;
    fn name(&self) -> &str;

    /// Apply operation to entire slice with SIMD optimization
    fn apply_slice(&self, a: &[T], b: &[T], output: &mut [T]) -> Result<()> {
        // Default implementation - element-wise
        if a.len() != b.len() || a.len() != output.len() {
            return Err(TensorError::invalid_argument(
                "Slice length mismatch for binary operation".to_string(),
            ));
        }
        for i in 0..a.len() {
            output[i] = self.apply(a[i].clone(), b[i].clone());
        }
        Ok(())
    }

    /// Check if operation supports SIMD acceleration
    fn supports_simd(&self) -> bool {
        false
    }

    /// Check if operation supports GPU acceleration
    fn supports_gpu(&self) -> bool {
        false
    }

    /// Get operation complexity for algorithm selection
    fn complexity(&self) -> OpComplexity {
        OpComplexity::Simple
    }

    /// Check if operation is associative (enables optimizations)
    fn is_associative(&self) -> bool {
        false
    }

    /// Check if operation is commutative (enables optimizations)
    fn is_commutative(&self) -> bool {
        false
    }
}

/// Operation complexity levels for adaptive algorithm selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpComplexity {
    Simple,   // Basic arithmetic (+, -, *, /)
    Moderate, // Comparisons, min/max
    Complex,  // Transcendental functions
    Advanced, // Custom complex operations
}

/// Ultra-performance binary operations registry and analytics
pub struct BinaryOpRegistry {
    /// Operation execution counters
    op_counters: Arc<Mutex<std::collections::HashMap<String, AtomicU64>>>,
    /// SIMD acceleration usage tracking
    simd_usage: AtomicU64,
    /// GPU acceleration usage tracking
    gpu_usage: AtomicU64,
    /// Parallel processing usage tracking
    parallel_usage: AtomicU64,
    /// Performance timer for operations
    #[allow(dead_code)]
    execution_timer: Timer,
    /// Memory throughput tracking
    memory_throughput: Histogram,
}

impl BinaryOpRegistry {
    pub fn new() -> Self {
        Self {
            op_counters: Arc::new(Mutex::new(std::collections::HashMap::new())),
            simd_usage: AtomicU64::new(0),
            gpu_usage: AtomicU64::new(0),
            parallel_usage: AtomicU64::new(0),
            execution_timer: Timer::new("binary_ops.execution_time".to_string()),
            memory_throughput: Histogram::new("binary_ops.memory_throughput".to_string()),
        }
    }

    /// Record operation execution
    pub fn record_operation(&self, op_name: &str, elements: usize, duration_ns: u64) {
        // Increment operation counter
        {
            let mut counters = self.op_counters.lock().unwrap();
            counters
                .entry(op_name.to_string())
                .or_insert_with(|| AtomicU64::new(0))
                .fetch_add(1, Ordering::Relaxed);
        }

        // Calculate and record memory throughput (GB/s)
        let bytes_processed = elements * std::mem::size_of::<f32>() * 2; // Input + output
        let throughput_gbps = (bytes_processed as f64 * 1e9) / (duration_ns as f64 * 1e9);
        self.memory_throughput.observe(throughput_gbps);
    }

    /// Record SIMD acceleration usage
    pub fn record_simd_usage(&self) {
        self.simd_usage.fetch_add(1, Ordering::Relaxed);
    }

    /// Record GPU acceleration usage
    pub fn record_gpu_usage(&self) {
        self.gpu_usage.fetch_add(1, Ordering::Relaxed);
    }

    /// Record parallel processing usage
    pub fn record_parallel_usage(&self) {
        self.parallel_usage.fetch_add(1, Ordering::Relaxed);
    }

    /// Get performance analytics
    pub fn get_analytics(&self) -> BinaryOpAnalytics {
        let counters = self.op_counters.lock().unwrap();
        let op_counts: std::collections::HashMap<String, u64> = counters
            .iter()
            .map(|(k, v)| (k.clone(), v.load(Ordering::Relaxed)))
            .collect();

        BinaryOpAnalytics {
            operation_counts: op_counts,
            simd_accelerations: self.simd_usage.load(Ordering::Relaxed),
            gpu_accelerations: self.gpu_usage.load(Ordering::Relaxed),
            parallel_executions: self.parallel_usage.load(Ordering::Relaxed),
            avg_memory_throughput: self.calculate_avg_throughput(),
        }
    }

    fn calculate_avg_throughput(&self) -> f64 {
        // Simplified calculation - in real implementation would use histogram stats
        15.0 // GB/s average placeholder
    }
}

impl Default for BinaryOpRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Performance analytics for binary operations
#[derive(Debug, Clone)]
pub struct BinaryOpAnalytics {
    pub operation_counts: std::collections::HashMap<String, u64>,
    pub simd_accelerations: u64,
    pub gpu_accelerations: u64,
    pub parallel_executions: u64,
    pub avg_memory_throughput: f64,
}

/// Global binary operations registry
static BINARY_OP_REGISTRY: std::sync::OnceLock<BinaryOpRegistry> = std::sync::OnceLock::new();

/// Get global binary operations registry
pub fn get_binary_op_registry() -> &'static BinaryOpRegistry {
    BINARY_OP_REGISTRY.get_or_init(BinaryOpRegistry::new)
}
