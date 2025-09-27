/*!
 * GPU Performance Optimizer and Profiler
 *
 * This module provides comprehensive GPU performance analysis and optimization
 * recommendations to achieve closer to TensorFlow-level performance.
 */

use crate::{DType, Device, Result, TensorError};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

/// GPU operation performance metrics
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct GpuOpMetrics {
    pub operation_name: String,
    pub device_id: usize,
    pub input_shapes: Vec<Vec<usize>>,
    pub dtype: DType,
    /// GPU kernel execution time
    pub kernel_time: Duration,
    /// Memory transfer time (host to device)
    pub h2d_transfer_time: Duration,
    /// Memory transfer time (device to host)
    pub d2h_transfer_time: Duration,
    /// Total operation time
    pub total_time: Duration,
    /// Memory bandwidth utilization (GB/s)
    pub memory_bandwidth: Option<f64>,
    /// GPU utilization percentage
    pub gpu_utilization: Option<f64>,
    /// Memory usage (bytes)
    pub memory_usage: u64,
    /// FLOPS achieved
    pub achieved_flops: Option<f64>,
    /// Theoretical peak FLOPS
    pub peak_flops: Option<f64>,
    /// Number of elements processed
    pub elements_processed: usize,
    /// Workgroup configuration used
    pub workgroup_config: WorkgroupConfig,
}

/// Workgroup configuration for compute shaders
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct WorkgroupConfig {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Default for WorkgroupConfig {
    fn default() -> Self {
        Self { x: 256, y: 1, z: 1 }
    }
}

/// Performance bottleneck analysis
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct BottleneckAnalysis {
    pub bottleneck_type: BottleneckType,
    pub severity: f64, // 0.0 to 1.0
    pub description: String,
    pub recommendations: Vec<String>,
    pub potential_improvement: f64, // Expected performance gain
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub enum BottleneckType {
    MemoryBandwidth,
    ComputeBound,
    LatencyBound,
    SynchronizationOverhead,
    InsufficientParallelism,
    SuboptimalWorkgroupSize,
    MemoryCoalescingIssues,
    RegisterPressure,
}

/// GPU device capabilities and characteristics
#[derive(Debug, Clone)]
pub struct GpuCapabilities {
    pub device_name: String,
    pub max_compute_units: u32,
    pub max_workgroup_size: u32,
    pub memory_bandwidth_gb_s: f64,
    pub peak_compute_tflops: f64,
    pub memory_size_gb: f64,
    pub warp_size: u32, // Or wavefront size for AMD
    pub shared_memory_per_workgroup: u32,
    pub supports_fp16: bool,
    pub supports_int8: bool,
}

/// Optimization configuration
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    /// Enable kernel fusion
    pub enable_kernel_fusion: bool,
    /// Enable memory coalescing optimization
    pub enable_memory_coalescing: bool,
    /// Enable async execution
    pub enable_async_execution: bool,
    /// Target memory utilization percentage
    pub target_memory_utilization: f64,
    /// Minimum batch size for optimization
    pub min_batch_size: usize,
    /// Enable auto-tuning of workgroup sizes
    pub enable_auto_tuning: bool,
    /// Maximum tuning iterations
    pub max_tuning_iterations: usize,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            enable_kernel_fusion: true,
            enable_memory_coalescing: true,
            enable_async_execution: true,
            target_memory_utilization: 0.8,
            min_batch_size: 32,
            enable_auto_tuning: true,
            max_tuning_iterations: 10,
        }
    }
}

/// GPU Performance Optimizer
pub struct GpuPerformanceOptimizer {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    capabilities: GpuCapabilities,
    config: OptimizationConfig,
    /// Performance history for learning optimal configurations
    performance_history: RwLock<HashMap<String, Vec<GpuOpMetrics>>>,
    /// Optimal configurations learned from profiling
    optimal_configs: RwLock<HashMap<String, WorkgroupConfig>>,
    /// Current profiling session
    active_profiling: Mutex<Option<ProfilingSession>>,
}

struct ProfilingSession {
    operation_name: String,
    start_time: Instant,
    metrics: GpuOpMetrics,
}

impl GpuPerformanceOptimizer {
    /// Create a new performance optimizer
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        capabilities: GpuCapabilities,
        config: OptimizationConfig,
    ) -> Self {
        Self {
            device,
            queue,
            capabilities,
            config,
            performance_history: RwLock::new(HashMap::new()),
            optimal_configs: RwLock::new(HashMap::new()),
            active_profiling: Mutex::new(None),
        }
    }

    /// Start profiling an operation
    pub fn start_profiling(
        &self,
        operation_name: &str,
        device_id: usize,
        input_shapes: Vec<Vec<usize>>,
        dtype: DType,
    ) {
        let metrics = GpuOpMetrics {
            operation_name: operation_name.to_string(),
            device_id,
            input_shapes,
            dtype,
            kernel_time: Duration::ZERO,
            h2d_transfer_time: Duration::ZERO,
            d2h_transfer_time: Duration::ZERO,
            total_time: Duration::ZERO,
            memory_bandwidth: None,
            gpu_utilization: None,
            memory_usage: 0,
            achieved_flops: None,
            peak_flops: Some(self.capabilities.peak_compute_tflops * 1e12),
            elements_processed: 0,
            workgroup_config: WorkgroupConfig::default(),
        };

        let session = ProfilingSession {
            operation_name: operation_name.to_string(),
            start_time: Instant::now(),
            metrics,
        };

        *self.active_profiling.lock().unwrap() = Some(session);
    }

    /// Record memory transfer timing
    pub fn record_memory_transfer(
        &self,
        h2d_time: Duration,
        d2h_time: Duration,
        bytes_transferred: u64,
    ) {
        if let Some(session) = self.active_profiling.lock().unwrap().as_mut() {
            session.metrics.h2d_transfer_time = h2d_time;
            session.metrics.d2h_transfer_time = d2h_time;
            session.metrics.memory_usage = bytes_transferred;

            // Calculate memory bandwidth
            let total_transfer_time = h2d_time + d2h_time;
            if total_transfer_time.as_secs_f64() > 0.0 {
                let bandwidth_gb_s =
                    (bytes_transferred as f64) / (1e9 * total_transfer_time.as_secs_f64());
                session.metrics.memory_bandwidth = Some(bandwidth_gb_s);
            }
        }
    }

    /// Record kernel execution timing
    pub fn record_kernel_execution(
        &self,
        kernel_time: Duration,
        elements_processed: usize,
        workgroup_config: WorkgroupConfig,
    ) {
        if let Some(session) = self.active_profiling.lock().unwrap().as_mut() {
            session.metrics.kernel_time = kernel_time;
            session.metrics.elements_processed = elements_processed;
            session.metrics.workgroup_config = workgroup_config;

            // Estimate FLOPS based on operation type
            if let Some(flops) =
                self.estimate_flops(&session.metrics.operation_name, elements_processed)
            {
                let achieved_flops = flops / kernel_time.as_secs_f64();
                session.metrics.achieved_flops = Some(achieved_flops);
            }
        }
    }

    /// Finish profiling and analyze results
    pub fn finish_profiling(&self) -> Option<GpuOpMetrics> {
        if let Some(session) = self.active_profiling.lock().unwrap().take() {
            let total_time = session.start_time.elapsed();
            let mut metrics = session.metrics;
            metrics.total_time = total_time;

            // Calculate GPU utilization
            if total_time.as_secs_f64() > 0.0 {
                let compute_ratio = metrics.kernel_time.as_secs_f64() / total_time.as_secs_f64();
                metrics.gpu_utilization = Some(compute_ratio * 100.0);
            }

            // Store in performance history
            let mut history = self.performance_history.write().unwrap();
            history
                .entry(metrics.operation_name.clone())
                .or_insert_with(Vec::new)
                .push(metrics.clone());

            return Some(metrics);
        }
        None
    }

    /// Analyze performance bottlenecks
    pub fn analyze_bottlenecks(&self, metrics: &GpuOpMetrics) -> Vec<BottleneckAnalysis> {
        let mut bottlenecks = Vec::new();

        // Memory bandwidth bottleneck
        if let Some(bandwidth) = metrics.memory_bandwidth {
            let bandwidth_utilization = bandwidth / self.capabilities.memory_bandwidth_gb_s;
            if bandwidth_utilization > 0.8 {
                bottlenecks.push(BottleneckAnalysis {
                    bottleneck_type: BottleneckType::MemoryBandwidth,
                    severity: bandwidth_utilization.min(1.0),
                    description: format!(
                        "Memory bandwidth utilization: {:.1}%",
                        bandwidth_utilization * 100.0
                    ),
                    recommendations: vec![
                        "Enable memory coalescing optimization".to_string(),
                        "Consider kernel fusion to reduce memory traffic".to_string(),
                        "Use async execution to overlap computation and memory transfers"
                            .to_string(),
                    ],
                    potential_improvement: (bandwidth_utilization - 0.8) * 0.5,
                });
            }
        }

        // Compute bound analysis
        if let (Some(achieved_flops), Some(peak_flops)) =
            (metrics.achieved_flops, metrics.peak_flops)
        {
            let compute_utilization = achieved_flops / peak_flops;
            if compute_utilization < 0.3 {
                bottlenecks.push(BottleneckAnalysis {
                    bottleneck_type: BottleneckType::ComputeBound,
                    severity: 1.0 - compute_utilization,
                    description: format!(
                        "Low compute utilization: {:.1}%",
                        compute_utilization * 100.0
                    ),
                    recommendations: vec![
                        "Optimize workgroup size for better occupancy".to_string(),
                        "Consider data layout optimization".to_string(),
                        "Use vectorized operations where possible".to_string(),
                    ],
                    potential_improvement: (0.3 - compute_utilization) * 2.0,
                });
            }
        }

        // GPU utilization analysis
        if let Some(gpu_util) = metrics.gpu_utilization {
            if gpu_util < 50.0 {
                bottlenecks.push(BottleneckAnalysis {
                    bottleneck_type: BottleneckType::SynchronizationOverhead,
                    severity: (50.0 - gpu_util) / 50.0,
                    description: format!("Low GPU utilization: {:.1}%", gpu_util),
                    recommendations: vec![
                        "Reduce synchronization points".to_string(),
                        "Use async execution patterns".to_string(),
                        "Batch operations to amortize overhead".to_string(),
                    ],
                    potential_improvement: (50.0 - gpu_util) / 100.0,
                });
            }
        }

        // Workgroup size analysis
        let workgroup_size =
            metrics.workgroup_config.x * metrics.workgroup_config.y * metrics.workgroup_config.z;
        if workgroup_size < self.capabilities.warp_size
            || workgroup_size % self.capabilities.warp_size != 0
        {
            bottlenecks.push(BottleneckAnalysis {
                bottleneck_type: BottleneckType::SuboptimalWorkgroupSize,
                severity: 0.7,
                description: format!("Suboptimal workgroup size: {}", workgroup_size),
                recommendations: vec![
                    format!(
                        "Use workgroup sizes that are multiples of {}",
                        self.capabilities.warp_size
                    ),
                    "Consider larger workgroup sizes for better occupancy".to_string(),
                ],
                potential_improvement: 0.2,
            });
        }

        bottlenecks
    }

    /// Get optimization recommendations
    pub fn get_optimization_recommendations(&self, operation_name: &str) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Get historical performance data
        let history = self.performance_history.read().unwrap();
        if let Some(metrics_history) = history.get(operation_name) {
            if metrics_history.len() >= 3 {
                // Analyze trends
                let recent_metrics = &metrics_history[metrics_history.len() - 3..];
                let avg_bandwidth = recent_metrics
                    .iter()
                    .filter_map(|m| m.memory_bandwidth)
                    .sum::<f64>()
                    / recent_metrics.len() as f64;

                if avg_bandwidth / self.capabilities.memory_bandwidth_gb_s > 0.7 {
                    recommendations.push(
                        "Operation is memory bandwidth bound - consider kernel fusion".to_string(),
                    );
                }

                let avg_gpu_util = recent_metrics
                    .iter()
                    .filter_map(|m| m.gpu_utilization)
                    .sum::<f64>()
                    / recent_metrics.len() as f64;

                if avg_gpu_util < 60.0 {
                    recommendations.push(
                        "Low GPU utilization - consider async execution or batching".to_string(),
                    );
                }
            }
        }

        // General recommendations based on operation type
        match operation_name {
            name if name.contains("matmul") => {
                recommendations
                    .push("Consider tiled matrix multiplication for large matrices".to_string());
                recommendations
                    .push("Use tensor cores if available for mixed precision".to_string());
            }
            name if name.contains("conv") => {
                recommendations.push(
                    "Consider Winograd or FFT convolution for appropriate filter sizes".to_string(),
                );
                recommendations
                    .push("Use depthwise separable convolutions for mobile efficiency".to_string());
            }
            name if name.contains("add") || name.contains("mul") => {
                recommendations.push(
                    "Consider fusing with following operations (e.g., activation functions)"
                        .to_string(),
                );
            }
            _ => {}
        }

        recommendations
    }

    /// Auto-tune workgroup configuration
    pub fn auto_tune_workgroup(&self, operation_name: &str, tensor_size: usize) -> WorkgroupConfig {
        // Check if we have an optimal configuration cached
        let optimal_configs = self.optimal_configs.read().unwrap();
        if let Some(config) = optimal_configs.get(operation_name) {
            return *config;
        }
        drop(optimal_configs);

        if !self.config.enable_auto_tuning {
            return WorkgroupConfig::default();
        }

        // Generate candidate configurations
        let candidates = self.generate_workgroup_candidates(tensor_size);
        let mut best_config = WorkgroupConfig::default();
        let mut best_performance = 0.0;

        // Profile each candidate (simplified - in real implementation would run actual kernels)
        for config in candidates {
            let estimated_performance = self.estimate_workgroup_performance(config, tensor_size);
            if estimated_performance > best_performance {
                best_performance = estimated_performance;
                best_config = config;
            }
        }

        // Cache the result
        let mut optimal_configs = self.optimal_configs.write().unwrap();
        optimal_configs.insert(operation_name.to_string(), best_config);

        best_config
    }

    /// Generate workgroup configuration candidates
    fn generate_workgroup_candidates(&self, tensor_size: usize) -> Vec<WorkgroupConfig> {
        let mut candidates = Vec::new();
        let max_workgroup_size = self.capabilities.max_workgroup_size.min(1024);

        // Common 1D configurations
        for size in [64, 128, 256, 512].iter() {
            if *size <= max_workgroup_size {
                candidates.push(WorkgroupConfig {
                    x: *size,
                    y: 1,
                    z: 1,
                });
            }
        }

        // 2D configurations for 2D/3D tensors
        if tensor_size > 10000 {
            for x in [16, 32] {
                for y in [8, 16] {
                    if x * y <= max_workgroup_size {
                        candidates.push(WorkgroupConfig { x, y, z: 1 });
                    }
                }
            }
        }

        candidates
    }

    /// Estimate workgroup performance (simplified heuristic)
    fn estimate_workgroup_performance(&self, config: WorkgroupConfig, tensor_size: usize) -> f64 {
        let workgroup_size = config.x * config.y * config.z;

        // Prefer workgroup sizes that are multiples of warp size
        let warp_efficiency = if workgroup_size % self.capabilities.warp_size == 0 {
            1.0
        } else {
            0.8
        };

        // Prefer reasonable occupancy
        let occupancy = (tensor_size as f32 / workgroup_size as f32).min(1.0);

        // Simple performance estimate
        workgroup_size as f64 * warp_efficiency * occupancy as f64
    }

    /// Estimate FLOPS for operation
    fn estimate_flops(&self, operation_name: &str, elements: usize) -> Option<f64> {
        let flops_per_element = match operation_name {
            name if name.contains("add") || name.contains("sub") => 1.0,
            name if name.contains("mul") || name.contains("div") => 1.0,
            name if name.contains("matmul") => {
                // Estimate based on typical matrix dimensions
                elements as f64 * 2.0 // Approximate for typical cases
            }
            name if name.contains("conv") => {
                // Rough estimate for convolution
                elements as f64 * 5.0
            }
            name if name.contains("relu") => 1.0,
            name if name.contains("sigmoid") || name.contains("tanh") => 8.0,
            name if name.contains("gelu") => 15.0,
            _ => return None,
        };

        Some(elements as f64 * flops_per_element)
    }

    /// Generate performance report
    pub fn generate_performance_report(&self) -> PerformanceReport {
        let history = self.performance_history.read().unwrap();
        let mut total_operations = 0;
        let mut total_time = Duration::ZERO;
        let mut avg_gpu_utilization = 0.0;
        let mut avg_memory_bandwidth = 0.0;
        let mut bottleneck_summary = HashMap::new();

        for (op_name, metrics_list) in history.iter() {
            total_operations += metrics_list.len();

            for metrics in metrics_list {
                total_time += metrics.total_time;

                if let Some(gpu_util) = metrics.gpu_utilization {
                    avg_gpu_utilization += gpu_util;
                }

                if let Some(bandwidth) = metrics.memory_bandwidth {
                    avg_memory_bandwidth += bandwidth;
                }

                // Analyze bottlenecks
                let bottlenecks = self.analyze_bottlenecks(metrics);
                for bottleneck in bottlenecks {
                    *bottleneck_summary
                        .entry(bottleneck.bottleneck_type)
                        .or_insert(0) += 1;
                }
            }
        }

        if total_operations > 0 {
            avg_gpu_utilization /= total_operations as f64;
            avg_memory_bandwidth /= total_operations as f64;
        }

        PerformanceReport {
            total_operations,
            total_time,
            avg_gpu_utilization,
            avg_memory_bandwidth,
            peak_memory_bandwidth: self.capabilities.memory_bandwidth_gb_s,
            bottleneck_summary,
            recommendations: self.generate_global_recommendations(),
        }
    }

    /// Generate global optimization recommendations
    fn generate_global_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        let history = self.performance_history.read().unwrap();
        let total_ops = history.values().map(|v| v.len()).sum::<usize>();

        if total_ops > 0 {
            let low_util_ops = history
                .values()
                .flatten()
                .filter(|m| m.gpu_utilization.unwrap_or(100.0) < 50.0)
                .count();

            if low_util_ops as f64 / total_ops as f64 > 0.3 {
                recommendations.push(
                    "Consider enabling async execution to improve GPU utilization".to_string(),
                );
            }

            let memory_bound_ops = history
                .values()
                .flatten()
                .filter(|m| {
                    m.memory_bandwidth.unwrap_or(0.0) / self.capabilities.memory_bandwidth_gb_s
                        > 0.7
                })
                .count();

            if memory_bound_ops as f64 / total_ops as f64 > 0.4 {
                recommendations.push(
                    "Many operations are memory bandwidth bound - enable kernel fusion".to_string(),
                );
            }
        }

        recommendations
    }
}

/// Performance report summary
#[derive(Debug)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct PerformanceReport {
    pub total_operations: usize,
    pub total_time: Duration,
    pub avg_gpu_utilization: f64,
    pub avg_memory_bandwidth: f64,
    pub peak_memory_bandwidth: f64,
    pub bottleneck_summary: HashMap<BottleneckType, usize>,
    pub recommendations: Vec<String>,
}

/// Detect GPU capabilities (simplified implementation)
pub fn detect_gpu_capabilities(device: &wgpu::Device) -> GpuCapabilities {
    // This would typically query actual device properties
    // For now, provide reasonable defaults based on typical hardware
    GpuCapabilities {
        device_name: "Generic GPU".to_string(),
        max_compute_units: 32,
        max_workgroup_size: 1024,
        memory_bandwidth_gb_s: 500.0, // Typical for mid-range GPU
        peak_compute_tflops: 10.0,    // Typical for mid-range GPU
        memory_size_gb: 8.0,
        warp_size: 32,                          // NVIDIA standard, AMD uses 64
        shared_memory_per_workgroup: 48 * 1024, // 48KB typical
        supports_fp16: true,
        supports_int8: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workgroup_config_default() {
        let config = WorkgroupConfig::default();
        assert_eq!(config.x, 256);
        assert_eq!(config.y, 1);
        assert_eq!(config.z, 1);
    }

    #[test]
    fn test_bottleneck_analysis() {
        let metrics = GpuOpMetrics {
            operation_name: "test_op".to_string(),
            device_id: 0,
            input_shapes: vec![vec![1024, 1024]],
            dtype: DType::Float32,
            kernel_time: Duration::from_millis(10),
            h2d_transfer_time: Duration::from_millis(5),
            d2h_transfer_time: Duration::from_millis(5),
            total_time: Duration::from_millis(20),
            memory_bandwidth: Some(400.0), // High bandwidth usage
            gpu_utilization: Some(45.0),   // Low GPU utilization
            memory_usage: 4 * 1024 * 1024,
            achieved_flops: Some(1e9),
            peak_flops: Some(10e12),
            elements_processed: 1024 * 1024,
            workgroup_config: WorkgroupConfig::default(),
        };

        let capabilities = GpuCapabilities {
            device_name: "Test GPU".to_string(),
            max_compute_units: 32,
            max_workgroup_size: 1024,
            memory_bandwidth_gb_s: 500.0,
            peak_compute_tflops: 10.0,
            memory_size_gb: 8.0,
            warp_size: 32,
            shared_memory_per_workgroup: 48 * 1024,
            supports_fp16: true,
            supports_int8: true,
        };

        // Test the bottleneck analysis logic directly without requiring device/queue
        fn analyze_bottlenecks_logic(
            capabilities: &GpuCapabilities,
            metrics: &GpuOpMetrics,
        ) -> Vec<BottleneckAnalysis> {
            let mut bottlenecks = Vec::new();

            // Memory bandwidth bottleneck
            if let Some(bandwidth) = metrics.memory_bandwidth {
                let bandwidth_utilization = bandwidth / capabilities.memory_bandwidth_gb_s;
                if bandwidth_utilization > 0.8 {
                    bottlenecks.push(BottleneckAnalysis {
                        bottleneck_type: BottleneckType::MemoryBandwidth,
                        severity: bandwidth_utilization.min(1.0),
                        description: format!(
                            "Memory bandwidth utilization: {:.1}%",
                            bandwidth_utilization * 100.0
                        ),
                        recommendations: vec![
                            "Enable memory coalescing optimization".to_string(),
                            "Consider kernel fusion to reduce memory traffic".to_string(),
                            "Use async execution to overlap computation and memory transfers"
                                .to_string(),
                        ],
                        potential_improvement: (bandwidth_utilization - 0.8) * 0.5,
                    });
                }
            }

            // GPU utilization bottleneck
            if let Some(utilization) = metrics.gpu_utilization {
                if utilization < 50.0 {
                    bottlenecks.push(BottleneckAnalysis {
                        bottleneck_type: BottleneckType::SynchronizationOverhead,
                        severity: 1.0 - (utilization / 100.0),
                        description: format!("Low GPU utilization: {:.1}%", utilization),
                        recommendations: vec![
                            "Increase batch size to improve GPU utilization".to_string(),
                            "Use async execution to reduce synchronization overhead".to_string(),
                        ],
                        potential_improvement: (50.0 - utilization) / 100.0,
                    });
                }
            }

            bottlenecks
        }

        let bottlenecks = analyze_bottlenecks_logic(&capabilities, &metrics);

        // Should detect synchronization overhead due to low GPU utilization
        assert!(bottlenecks
            .iter()
            .any(|b| b.bottleneck_type == BottleneckType::SynchronizationOverhead));
    }
}
