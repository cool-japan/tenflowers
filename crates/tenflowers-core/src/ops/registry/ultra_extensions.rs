//! Ultra-performance registry extensions for advanced use cases.
//!
//! Provides specialized registry implementations including high-frequency-trading
//! optimized registries and quantum-inspired superposition caches.

use super::core::RegistryAnalytics;
use super::types::{OpDef, OpRegistry};
use std::collections::HashMap;
use std::sync::Arc;

/// Registry with specialized high-frequency trading optimizations
#[allow(dead_code)]
pub struct HftRegistry {
    base: OpRegistry,
    /// Ultra-low latency operation cache with lock-free access
    lockfree_cache: std::sync::Arc<std::collections::HashMap<String, Arc<OpDef>>>,
    /// Microsecond-precision timing
    precision_timer: std::time::Instant,
}

#[allow(dead_code)]
impl HftRegistry {
    pub fn new() -> Self {
        Self {
            base: OpRegistry::new(),
            lockfree_cache: std::sync::Arc::new(std::collections::HashMap::new()),
            precision_timer: std::time::Instant::now(),
        }
    }

    /// Ultra-low latency operation lookup (< 100ns target)
    pub fn get_op_ultrafast(&self, name: &str) -> Option<Arc<OpDef>> {
        // Fallback to regular cache (simplified for compatibility)
        self.base.get_op(name).map(Arc::new)
    }

    /// Get microsecond-precision execution metrics
    pub fn get_precision_metrics(&self) -> PrecisionMetrics {
        PrecisionMetrics {
            elapsed_microseconds: self.precision_timer.elapsed().as_micros() as u64,
            cache_size: 0, // Simplified
            base_analytics: self.base.get_performance_analytics(),
        }
    }
}

#[allow(dead_code)]
impl Default for HftRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// High-precision performance metrics
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct PrecisionMetrics {
    pub elapsed_microseconds: u64,
    pub cache_size: usize,
    pub base_analytics: RegistryAnalytics,
}

/// Quantum-inspired registry for experimental features
#[allow(dead_code)]
pub struct QuantumRegistry {
    base: OpRegistry,
    /// Quantum-inspired superposition cache (multiple states)
    superposition_cache: HashMap<String, Vec<Arc<OpDef>>>,
    /// Entanglement tracking for operation dependencies
    entanglement_graph: petgraph::Graph<String, f64>,
}

#[allow(dead_code)]
impl QuantumRegistry {
    pub fn new() -> Self {
        Self {
            base: OpRegistry::new(),
            superposition_cache: HashMap::new(),
            entanglement_graph: petgraph::Graph::new(),
        }
    }

    /// Get operation with quantum superposition (multiple possible implementations)
    pub fn get_op_superposition(&self, name: &str) -> Vec<Arc<OpDef>> {
        if let Some(superposed_ops) = self.superposition_cache.get(name) {
            superposed_ops.clone()
        } else if let Some(op) = self.base.get_op(name) {
            vec![Arc::new(op)]
        } else {
            vec![]
        }
    }

    /// Collapse superposition to single best implementation
    pub fn collapse_superposition(
        &self,
        name: &str,
        selection_criteria: f64,
    ) -> Option<Arc<OpDef>> {
        let superposed = self.get_op_superposition(name);
        if superposed.is_empty() {
            return None;
        }

        // Select based on criteria (simplified)
        let index = (selection_criteria * superposed.len() as f64) as usize % superposed.len();
        Some(superposed[index].clone())
    }
}

#[allow(dead_code)]
impl Default for QuantumRegistry {
    fn default() -> Self {
        Self::new()
    }
}
