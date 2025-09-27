//! Performance Bottleneck Detection for TenfloweRS
//!
//! This module provides intelligent bottleneck detection and performance
//! optimization recommendations, addressing the TODO item for future enhancement.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;
use std::time::Instant;

/// Python-accessible bottleneck detector
#[pyclass]
pub struct PyBottleneckDetector {
    inner: BottleneckDetector,
}

/// Core bottleneck detection logic
pub struct BottleneckDetector {
    performance_history: Vec<OperationProfile>,
    bottleneck_thresholds: BottleneckThresholds,
    analysis_cache: HashMap<String, BottleneckAnalysis>,
}

/// Performance profile for a single operation
#[derive(Debug, Clone)]
pub struct OperationProfile {
    pub operation_name: String,
    pub execution_time_ms: f64,
    pub memory_usage_mb: f64,
    pub gpu_utilization: f64,
    pub cpu_utilization: f64,
    pub memory_bandwidth_utilization: f64,
    pub input_size: usize,
    pub batch_size: usize,
    pub timestamp: Instant,
}

/// Thresholds for bottleneck detection
#[derive(Debug, Clone)]
pub struct BottleneckThresholds {
    pub slow_operation_ms: f64,
    pub high_memory_usage_mb: f64,
    pub low_gpu_utilization: f64,
    pub low_cpu_utilization: f64,
    pub low_memory_bandwidth: f64,
    pub regression_threshold: f64, // Percentage increase considered a regression
}

impl Default for BottleneckThresholds {
    fn default() -> Self {
        Self {
            slow_operation_ms: 100.0,     // Operations > 100ms are slow
            high_memory_usage_mb: 1000.0, // > 1GB memory usage is high
            low_gpu_utilization: 50.0,    // < 50% GPU utilization is low
            low_cpu_utilization: 30.0,    // < 30% CPU utilization is low
            low_memory_bandwidth: 40.0,   // < 40% memory bandwidth utilization is low
            regression_threshold: 20.0,   // > 20% slower is a regression
        }
    }
}

/// Bottleneck analysis result
#[derive(Debug, Clone)]
pub struct BottleneckAnalysis {
    pub bottleneck_type: BottleneckType,
    pub severity: BottleneckSeverity,
    pub affected_operations: Vec<String>,
    pub performance_impact: f64, // Percentage of total execution time
    pub recommendations: Vec<String>,
    pub confidence_score: f64, // 0.0 to 1.0
}

/// Types of performance bottlenecks
#[derive(Debug, Clone, PartialEq)]
pub enum BottleneckType {
    CpuBound,
    MemoryBound,
    GpuUnderutilized,
    MemoryBandwidthLimited,
    ComputeBound,
    IoWait,
    Serialization,
    DataTransfer,
    AlgorithmicInefficiency,
}

/// Severity levels for bottlenecks
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum BottleneckSeverity {
    Low,
    Medium,
    High,
    Critical,
}

impl BottleneckDetector {
    pub fn new() -> Self {
        Self {
            performance_history: Vec::new(),
            bottleneck_thresholds: BottleneckThresholds::default(),
            analysis_cache: HashMap::new(),
        }
    }

    /// Record a performance profile for analysis
    pub fn record_operation(&mut self, profile: OperationProfile) {
        self.performance_history.push(profile);

        // Keep only recent history (last 1000 operations)
        if self.performance_history.len() > 1000 {
            self.performance_history.drain(0..500);
        }

        // Clear cache when new data arrives
        self.analysis_cache.clear();
    }

    /// Detect bottlenecks in the current performance profile
    pub fn detect_bottlenecks(&mut self) -> Vec<BottleneckAnalysis> {
        let mut bottlenecks = Vec::new();

        // Check for different types of bottlenecks
        if let Some(cpu_bottleneck) = self.detect_cpu_bottleneck() {
            bottlenecks.push(cpu_bottleneck);
        }

        if let Some(memory_bottleneck) = self.detect_memory_bottleneck() {
            bottlenecks.push(memory_bottleneck);
        }

        if let Some(gpu_bottleneck) = self.detect_gpu_underutilization() {
            bottlenecks.push(gpu_bottleneck);
        }

        if let Some(bandwidth_bottleneck) = self.detect_memory_bandwidth_bottleneck() {
            bottlenecks.push(bandwidth_bottleneck);
        }

        if let Some(algo_bottleneck) = self.detect_algorithmic_inefficiency() {
            bottlenecks.push(algo_bottleneck);
        }

        if let Some(regression) = self.detect_performance_regression() {
            bottlenecks.push(regression);
        }

        // Sort by severity and impact
        bottlenecks.sort_by(|a, b| {
            b.severity
                .partial_cmp(&a.severity)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(
                    b.performance_impact
                        .partial_cmp(&a.performance_impact)
                        .unwrap_or(std::cmp::Ordering::Equal),
                )
        });

        bottlenecks
    }

    /// Detect CPU-bound bottlenecks
    fn detect_cpu_bottleneck(&self) -> Option<BottleneckAnalysis> {
        let recent_ops: Vec<_> = self.performance_history.iter().rev().take(50).collect();

        if recent_ops.is_empty() {
            return None;
        }

        let avg_cpu_utilization: f64 =
            recent_ops.iter().map(|op| op.cpu_utilization).sum::<f64>() / recent_ops.len() as f64;

        let slow_cpu_ops: Vec<_> = recent_ops
            .iter()
            .filter(|op| op.cpu_utilization > 80.0 && op.gpu_utilization < 50.0)
            .map(|op| op.operation_name.clone())
            .collect();

        if avg_cpu_utilization > 85.0 && !slow_cpu_ops.is_empty() {
            let severity = if avg_cpu_utilization > 95.0 {
                BottleneckSeverity::Critical
            } else if avg_cpu_utilization > 90.0 {
                BottleneckSeverity::High
            } else {
                BottleneckSeverity::Medium
            };

            let recommendations = vec![
                "Consider moving operations to GPU for parallel execution".to_string(),
                "Optimize CPU algorithms with SIMD or vectorization".to_string(),
                "Reduce CPU-intensive preprocessing steps".to_string(),
                "Use multi-threading for CPU-bound operations".to_string(),
            ];

            return Some(BottleneckAnalysis {
                bottleneck_type: BottleneckType::CpuBound,
                severity,
                affected_operations: slow_cpu_ops,
                performance_impact: (avg_cpu_utilization - 50.0) / 50.0 * 100.0,
                recommendations,
                confidence_score: 0.8,
            });
        }

        None
    }

    /// Detect memory-bound bottlenecks
    fn detect_memory_bottleneck(&self) -> Option<BottleneckAnalysis> {
        let recent_ops: Vec<_> = self.performance_history.iter().rev().take(50).collect();

        if recent_ops.is_empty() {
            return None;
        }

        let high_memory_ops: Vec<_> = recent_ops
            .iter()
            .filter(|op| op.memory_usage_mb > self.bottleneck_thresholds.high_memory_usage_mb)
            .map(|op| op.operation_name.clone())
            .collect();

        let avg_memory_usage: f64 =
            recent_ops.iter().map(|op| op.memory_usage_mb).sum::<f64>() / recent_ops.len() as f64;

        if avg_memory_usage > self.bottleneck_thresholds.high_memory_usage_mb
            && !high_memory_ops.is_empty()
        {
            let severity = if avg_memory_usage > 4000.0 {
                BottleneckSeverity::Critical
            } else if avg_memory_usage > 2000.0 {
                BottleneckSeverity::High
            } else {
                BottleneckSeverity::Medium
            };

            let recommendations = vec![
                "Reduce batch size to lower memory usage".to_string(),
                "Use gradient checkpointing for large models".to_string(),
                "Enable mixed precision training (FP16/BF16)".to_string(),
                "Implement memory pooling and reuse strategies".to_string(),
                "Consider model sharding for very large models".to_string(),
            ];

            return Some(BottleneckAnalysis {
                bottleneck_type: BottleneckType::MemoryBound,
                severity,
                affected_operations: high_memory_ops,
                performance_impact: (avg_memory_usage / 1000.0).min(100.0),
                recommendations,
                confidence_score: 0.9,
            });
        }

        None
    }

    /// Detect GPU underutilization
    fn detect_gpu_underutilization(&self) -> Option<BottleneckAnalysis> {
        let recent_ops: Vec<_> = self.performance_history.iter().rev().take(50).collect();

        if recent_ops.is_empty() {
            return None;
        }

        let avg_gpu_utilization: f64 =
            recent_ops.iter().map(|op| op.gpu_utilization).sum::<f64>() / recent_ops.len() as f64;

        let underutilized_ops: Vec<_> = recent_ops
            .iter()
            .filter(|op| op.gpu_utilization < self.bottleneck_thresholds.low_gpu_utilization)
            .map(|op| op.operation_name.clone())
            .collect();

        if avg_gpu_utilization < self.bottleneck_thresholds.low_gpu_utilization
            && !underutilized_ops.is_empty()
        {
            let severity = if avg_gpu_utilization < 20.0 {
                BottleneckSeverity::High
            } else if avg_gpu_utilization < 35.0 {
                BottleneckSeverity::Medium
            } else {
                BottleneckSeverity::Low
            };

            let recommendations = vec![
                "Increase batch size to better utilize GPU parallelism".to_string(),
                "Use GPU-optimized operations instead of CPU fallbacks".to_string(),
                "Minimize data transfers between CPU and GPU".to_string(),
                "Consider kernel fusion to reduce GPU kernel launch overhead".to_string(),
                "Profile GPU memory bandwidth utilization".to_string(),
            ];

            return Some(BottleneckAnalysis {
                bottleneck_type: BottleneckType::GpuUnderutilized,
                severity,
                affected_operations: underutilized_ops,
                performance_impact: (self.bottleneck_thresholds.low_gpu_utilization
                    - avg_gpu_utilization)
                    / self.bottleneck_thresholds.low_gpu_utilization
                    * 100.0,
                recommendations,
                confidence_score: 0.85,
            });
        }

        None
    }

    /// Detect memory bandwidth bottlenecks
    fn detect_memory_bandwidth_bottleneck(&self) -> Option<BottleneckAnalysis> {
        let recent_ops: Vec<_> = self.performance_history.iter().rev().take(50).collect();

        if recent_ops.is_empty() {
            return None;
        }

        let avg_bandwidth_utilization: f64 = recent_ops
            .iter()
            .map(|op| op.memory_bandwidth_utilization)
            .sum::<f64>()
            / recent_ops.len() as f64;

        let bandwidth_limited_ops: Vec<_> = recent_ops
            .iter()
            .filter(|op| op.memory_bandwidth_utilization > 85.0 && op.gpu_utilization < 70.0)
            .map(|op| op.operation_name.clone())
            .collect();

        if avg_bandwidth_utilization > 85.0 && !bandwidth_limited_ops.is_empty() {
            let recommendations = vec![
                "Use higher compute-to-memory ratio operations".to_string(),
                "Implement data layout optimizations (e.g., tiling)".to_string(),
                "Reduce precision (FP16/BF16) to decrease memory bandwidth".to_string(),
                "Use compression techniques for data transfer".to_string(),
                "Optimize memory access patterns for cache efficiency".to_string(),
            ];

            return Some(BottleneckAnalysis {
                bottleneck_type: BottleneckType::MemoryBandwidthLimited,
                severity: BottleneckSeverity::High,
                affected_operations: bandwidth_limited_ops,
                performance_impact: (avg_bandwidth_utilization - 50.0) / 50.0 * 100.0,
                recommendations,
                confidence_score: 0.75,
            });
        }

        None
    }

    /// Detect algorithmic inefficiencies
    fn detect_algorithmic_inefficiency(&self) -> Option<BottleneckAnalysis> {
        let recent_ops: Vec<_> = self.performance_history.iter().rev().take(100).collect();

        if recent_ops.len() < 20 {
            return None;
        }

        // Look for operations with poor scaling characteristics
        let mut inefficient_ops = Vec::new();
        let operation_groups: HashMap<String, Vec<&OperationProfile>> =
            recent_ops.iter().fold(HashMap::new(), |mut acc, op| {
                acc.entry(op.operation_name.clone()).or_default().push(op);
                acc
            });

        for (op_name, profiles) in operation_groups {
            if profiles.len() < 5 {
                continue;
            }

            // Check if execution time scales poorly with input size
            let correlations = self.calculate_scaling_correlation(&profiles);
            if correlations.time_vs_size > 2.0 {
                // Super-linear scaling
                inefficient_ops.push(op_name);
            }
        }

        if !inefficient_ops.is_empty() {
            let recommendations = vec![
                "Review algorithm complexity and consider more efficient alternatives".to_string(),
                "Use optimized libraries (BLAS, cuDNN) for standard operations".to_string(),
                "Implement algorithmic optimizations (early stopping, pruning)".to_string(),
                "Consider approximate algorithms for less critical operations".to_string(),
                "Profile specific algorithm hot spots for targeted optimization".to_string(),
            ];

            return Some(BottleneckAnalysis {
                bottleneck_type: BottleneckType::AlgorithmicInefficiency,
                severity: BottleneckSeverity::Medium,
                affected_operations: inefficient_ops,
                performance_impact: 30.0, // Estimated
                recommendations,
                confidence_score: 0.65,
            });
        }

        None
    }

    /// Detect performance regressions
    fn detect_performance_regression(&self) -> Option<BottleneckAnalysis> {
        if self.performance_history.len() < 100 {
            return None;
        }

        let recent_ops: Vec<_> = self.performance_history.iter().rev().take(50).collect();

        let historical_ops: Vec<_> = self
            .performance_history
            .iter()
            .rev()
            .skip(50)
            .take(50)
            .collect();

        let recent_avg_time: f64 = recent_ops
            .iter()
            .map(|op| op.execution_time_ms)
            .sum::<f64>()
            / recent_ops.len() as f64;

        let historical_avg_time: f64 = historical_ops
            .iter()
            .map(|op| op.execution_time_ms)
            .sum::<f64>()
            / historical_ops.len() as f64;

        let regression_percentage =
            ((recent_avg_time - historical_avg_time) / historical_avg_time) * 100.0;

        if regression_percentage > self.bottleneck_thresholds.regression_threshold {
            let severity = if regression_percentage > 50.0 {
                BottleneckSeverity::Critical
            } else if regression_percentage > 30.0 {
                BottleneckSeverity::High
            } else {
                BottleneckSeverity::Medium
            };

            let recommendations = vec![
                format!(
                    "Performance regression detected: {:.1}% slower than baseline",
                    regression_percentage
                ),
                "Review recent code changes for performance impact".to_string(),
                "Run detailed profiling to identify regression source".to_string(),
                "Consider reverting to previous configuration if critical".to_string(),
                "Implement performance regression tests in CI/CD".to_string(),
            ];

            return Some(BottleneckAnalysis {
                bottleneck_type: BottleneckType::AlgorithmicInefficiency,
                severity,
                affected_operations: vec!["Multiple operations".to_string()],
                performance_impact: regression_percentage,
                recommendations,
                confidence_score: 0.9,
            });
        }

        None
    }

    /// Calculate scaling correlation for an operation
    fn calculate_scaling_correlation(&self, profiles: &[&OperationProfile]) -> ScalingCorrelation {
        if profiles.len() < 3 {
            return ScalingCorrelation { time_vs_size: 1.0 };
        }

        // Simple correlation calculation (could be enhanced with proper statistical methods)
        let mut size_time_pairs: Vec<_> = profiles
            .iter()
            .map(|p| (p.input_size as f64, p.execution_time_ms))
            .collect();

        size_time_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let first = &size_time_pairs[0];
        let last = &size_time_pairs[size_time_pairs.len() - 1];

        let size_ratio = last.0 / first.0;
        let time_ratio = last.1 / first.1;

        let scaling_factor = if size_ratio > 1.0 {
            time_ratio / size_ratio
        } else {
            1.0
        };

        ScalingCorrelation {
            time_vs_size: scaling_factor,
        }
    }

    /// Generate comprehensive optimization recommendations
    pub fn generate_optimization_recommendations(
        &self,
        bottlenecks: &[BottleneckAnalysis],
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        if bottlenecks.is_empty() {
            recommendations
                .push("No significant bottlenecks detected. Performance looks good!".to_string());
            recommendations.push(
                "Consider running workload with larger batch sizes to identify scalability limits"
                    .to_string(),
            );
            return recommendations;
        }

        // Prioritize recommendations based on bottleneck severity and impact
        for bottleneck in bottlenecks.iter().take(3) {
            // Top 3 bottlenecks
            recommendations.push(format!(
                "🔴 {} bottleneck detected (Severity: {:?}, Impact: {:.1}%)",
                format!("{:?}", bottleneck.bottleneck_type).replace('_', " "),
                bottleneck.severity,
                bottleneck.performance_impact
            ));

            for rec in &bottleneck.recommendations {
                recommendations.push(format!("  • {}", rec));
            }
            recommendations.push("".to_string()); // Separator
        }

        // General recommendations
        recommendations.push("📊 General Performance Tips:".to_string());
        recommendations.push("  • Use profiling tools to validate optimization impact".to_string());
        recommendations.push("  • Implement performance monitoring in production".to_string());
        recommendations
            .push("  • Consider A/B testing different optimization strategies".to_string());

        recommendations
    }
}

/// Scaling correlation metrics
#[derive(Debug)]
struct ScalingCorrelation {
    time_vs_size: f64,
}

impl Default for BottleneckDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for PyBottleneckDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[pymethods]
impl PyBottleneckDetector {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: BottleneckDetector::new(),
        }
    }

    /// Record operation performance data
    pub fn record_operation(
        &mut self,
        operation_name: String,
        execution_time_ms: f64,
        memory_usage_mb: f64,
        gpu_utilization: f64,
        cpu_utilization: f64,
        memory_bandwidth_utilization: f64,
        input_size: usize,
        batch_size: usize,
    ) {
        let profile = OperationProfile {
            operation_name,
            execution_time_ms,
            memory_usage_mb,
            gpu_utilization,
            cpu_utilization,
            memory_bandwidth_utilization,
            input_size,
            batch_size,
            timestamp: Instant::now(),
        };

        self.inner.record_operation(profile);
    }

    /// Detect current bottlenecks
    pub fn detect_bottlenecks(&mut self, py: Python) -> PyResult<PyObject> {
        let bottlenecks = self.inner.detect_bottlenecks();
        let py_list = PyList::empty(py);

        for bottleneck in bottlenecks {
            let py_dict = PyDict::new(py);
            py_dict.set_item("type", format!("{:?}", bottleneck.bottleneck_type))?;
            py_dict.set_item("severity", format!("{:?}", bottleneck.severity))?;
            py_dict.set_item(
                "affected_operations",
                PyList::new(py, bottleneck.affected_operations)?,
            )?;
            py_dict.set_item("performance_impact", bottleneck.performance_impact)?;
            py_dict.set_item(
                "recommendations",
                PyList::new(py, bottleneck.recommendations)?,
            )?;
            py_dict.set_item("confidence_score", bottleneck.confidence_score)?;

            py_list.append(py_dict)?;
        }

        Ok(py_list.into())
    }

    /// Get optimization recommendations
    pub fn get_optimization_recommendations(&mut self) -> PyResult<Vec<String>> {
        let bottlenecks = self.inner.detect_bottlenecks();
        Ok(self
            .inner
            .generate_optimization_recommendations(&bottlenecks))
    }

    /// Get performance summary
    pub fn get_performance_summary(&self, py: Python) -> PyResult<PyObject> {
        let py_dict = PyDict::new(py);

        if self.inner.performance_history.is_empty() {
            py_dict.set_item("total_operations", 0)?;
            return Ok(py_dict.into());
        }

        let recent_ops: Vec<_> = self
            .inner
            .performance_history
            .iter()
            .rev()
            .take(50)
            .collect();

        let avg_execution_time: f64 = recent_ops
            .iter()
            .map(|op| op.execution_time_ms)
            .sum::<f64>()
            / recent_ops.len() as f64;

        let avg_memory_usage: f64 =
            recent_ops.iter().map(|op| op.memory_usage_mb).sum::<f64>() / recent_ops.len() as f64;

        let avg_gpu_utilization: f64 =
            recent_ops.iter().map(|op| op.gpu_utilization).sum::<f64>() / recent_ops.len() as f64;

        py_dict.set_item("total_operations", self.inner.performance_history.len())?;
        py_dict.set_item("avg_execution_time_ms", avg_execution_time)?;
        py_dict.set_item("avg_memory_usage_mb", avg_memory_usage)?;
        py_dict.set_item("avg_gpu_utilization", avg_gpu_utilization)?;

        Ok(py_dict.into())
    }
}

/// Register bottleneck detection functions with Python module
pub fn register_bottleneck_detection_functions(
    _py: Python,
    m: &Bound<'_, PyModule>,
) -> PyResult<()> {
    m.add_class::<PyBottleneckDetector>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bottleneck_detector_creation() {
        let detector = BottleneckDetector::new();
        assert_eq!(detector.performance_history.len(), 0);
    }

    #[test]
    fn test_operation_recording() {
        let mut detector = BottleneckDetector::new();
        let profile = OperationProfile {
            operation_name: "test_op".to_string(),
            execution_time_ms: 100.0,
            memory_usage_mb: 500.0,
            gpu_utilization: 75.0,
            cpu_utilization: 25.0,
            memory_bandwidth_utilization: 60.0,
            input_size: 1000,
            batch_size: 32,
            timestamp: Instant::now(),
        };

        detector.record_operation(profile);
        assert_eq!(detector.performance_history.len(), 1);
    }

    #[test]
    fn test_cpu_bottleneck_detection() {
        let mut detector = BottleneckDetector::new();

        // Add operations with high CPU utilization
        for i in 0..20 {
            let profile = OperationProfile {
                operation_name: format!("cpu_heavy_{}", i),
                execution_time_ms: 200.0,
                memory_usage_mb: 100.0,
                gpu_utilization: 20.0,
                cpu_utilization: 95.0,
                memory_bandwidth_utilization: 30.0,
                input_size: 1000,
                batch_size: 16,
                timestamp: Instant::now(),
            };
            detector.record_operation(profile);
        }

        let bottlenecks = detector.detect_bottlenecks();
        assert!(!bottlenecks.is_empty());
        assert!(bottlenecks
            .iter()
            .any(|b| b.bottleneck_type == BottleneckType::CpuBound));
    }
}
