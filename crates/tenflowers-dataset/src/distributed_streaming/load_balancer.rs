//! Load balancing strategies for distributed streaming

use super::types::{PartitionStrategy, WorkerMetrics, WorkerStatus};

/// Load balancing policy
#[derive(Debug, Clone)]
pub enum LoadBalancingPolicy {
    /// Static partitioning (default)
    Static,
    /// Work-stealing: idle workers steal from slow ones
    WorkStealing { steal_threshold: f64 },
    /// Proportional: assign proportionally to throughput
    Proportional,
}

impl Default for LoadBalancingPolicy {
    fn default() -> Self {
        Self::Static
    }
}

/// Compute load imbalance metric (coefficient of variation of throughputs)
pub fn compute_load_imbalance(metrics: &[WorkerMetrics]) -> f64 {
    if metrics.is_empty() {
        return 0.0;
    }

    let total: f64 = metrics.iter().map(|m| m.throughput_samples_per_sec).sum();
    let n = metrics.len() as f64;
    let mean = total / n;

    if mean <= 0.0 {
        return 0.0;
    }

    let variance: f64 = metrics
        .iter()
        .map(|m| {
            let diff = m.throughput_samples_per_sec - mean;
            diff * diff
        })
        .sum::<f64>()
        / n;

    variance.sqrt() / mean
}

/// Determine which workers need more work assigned
pub fn find_underloaded_workers(metrics: &[WorkerMetrics], threshold: f64) -> Vec<usize> {
    let imbalance = compute_load_imbalance(metrics);
    if imbalance <= threshold {
        return Vec::new();
    }

    let total: f64 = metrics.iter().map(|m| m.throughput_samples_per_sec).sum();
    let n = metrics.len() as f64;
    let mean = total / n;

    metrics
        .iter()
        .filter(|m| m.throughput_samples_per_sec > mean)
        .map(|m| m.rank)
        .collect()
}

/// Determine which workers are overloaded
pub fn find_overloaded_workers(metrics: &[WorkerMetrics], threshold: f64) -> Vec<usize> {
    let imbalance = compute_load_imbalance(metrics);
    if imbalance <= threshold {
        return Vec::new();
    }

    let total: f64 = metrics.iter().map(|m| m.throughput_samples_per_sec).sum();
    let n = metrics.len() as f64;
    let mean = total / n;

    metrics
        .iter()
        .filter(|m| m.throughput_samples_per_sec < mean)
        .map(|m| m.rank)
        .collect()
}

/// Compute proportional index redistribution weights based on throughput
pub fn compute_redistribution_weights(metrics: &[WorkerMetrics]) -> Vec<f64> {
    let total: f64 = metrics.iter().map(|m| m.throughput_samples_per_sec).sum();

    if total <= 0.0 {
        let n = metrics.len();
        return vec![1.0 / n as f64; n];
    }

    metrics
        .iter()
        .map(|m| m.throughput_samples_per_sec / total)
        .collect()
}
