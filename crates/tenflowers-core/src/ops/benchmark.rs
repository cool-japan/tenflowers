//! Kernel Benchmarking Framework
//!
//! This module provides a comprehensive benchmarking system for measuring
//! and comparing the performance of tensor operations across different
//! devices, data types, and configurations.

use super::registry::{OpVersion, OP_REGISTRY};
use crate::{DType, Device, Result, Shape, Tensor, TensorError};
use std::collections::HashMap;
use std::sync::RwLock;
use std::time::{Duration, Instant};

#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

/// Benchmark result for a single operation run
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct BenchmarkResult {
    /// Operation name
    pub operation: String,
    /// Operation version
    pub version: OpVersion,
    /// Device used
    pub device: Device,
    /// Data type used
    pub dtype: DType,
    /// Input shapes
    pub input_shapes: Vec<Shape>,
    /// Execution time
    pub duration: Duration,
    /// Memory allocated during operation (bytes)
    pub memory_used: Option<u64>,
    /// FLOPS (floating-point operations per second)
    pub flops: Option<f64>,
    /// Throughput (elements per second)
    pub throughput: Option<f64>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Benchmark configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct BenchmarkConfig {
    /// Number of warmup iterations
    pub warmup_iterations: usize,
    /// Number of measurement iterations
    pub measurement_iterations: usize,
    /// Whether to measure memory usage
    pub measure_memory: bool,
    /// Whether to calculate FLOPS
    pub calculate_flops: bool,
    /// Minimum execution time for reliable measurements
    pub min_execution_time: Duration,
    /// Maximum execution time before timeout
    pub max_execution_time: Duration,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 10,
            measurement_iterations: 100,
            measure_memory: true,
            calculate_flops: true,
            min_execution_time: Duration::from_millis(1),
            max_execution_time: Duration::from_secs(30),
        }
    }
}

/// Regression detection report
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct RegressionReport {
    pub operation: String,
    pub device: Device,
    pub dtype: DType,
    pub input_shapes: Vec<Shape>,
    pub current_duration: Duration,
    pub baseline_duration: Duration,
    pub duration_regression: f64, // Ratio - 1.0 (e.g., 0.5 = 50% slower)
    pub current_memory: Option<u64>,
    pub baseline_memory: Option<u64>,
    pub memory_regression: Option<f64>,
    pub severity: RegressionSeverity,
}

/// Severity level for performance regressions
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub enum RegressionSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Statistical summary for a set of measurements
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct Statistics {
    pub mean: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub median: f64,
    pub count: usize,
}

/// Comprehensive benchmark statistics
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct BenchmarkStatistics {
    pub total_benchmarks: usize,
    pub duration_stats: Option<Statistics>,
    pub throughput_stats: Option<Statistics>,
    pub memory_stats: Option<Statistics>,
    pub flops_stats: Option<Statistics>,
}

/// Helper for computing statistical summaries
struct StatisticalSummary {
    samples: Vec<f64>,
}

impl StatisticalSummary {
    fn new() -> Self {
        Self {
            samples: Vec::new(),
        }
    }

    fn add_sample(&mut self, value: f64) {
        self.samples.push(value);
    }

    fn finalize(mut self) -> Option<Statistics> {
        if self.samples.is_empty() {
            return None;
        }

        self.samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let count = self.samples.len();
        let sum: f64 = self.samples.iter().sum();
        let mean = sum / count as f64;

        let variance = if count > 1 {
            self.samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (count - 1) as f64
        // Sample standard deviation
        } else {
            0.0 // Single sample case
        };
        let std_dev = variance.sqrt();

        let min = self.samples[0];
        let max = self.samples[count - 1];
        let median = if count % 2 == 0 {
            (self.samples[count / 2 - 1] + self.samples[count / 2]) / 2.0
        } else {
            self.samples[count / 2]
        };

        Some(Statistics {
            mean,
            std_dev,
            min,
            max,
            median,
            count,
        })
    }
}

/// Benchmark suite for systematic performance testing
pub struct BenchmarkSuite {
    /// Configuration for benchmarks
    config: BenchmarkConfig,
    /// Collected benchmark results
    results: RwLock<Vec<BenchmarkResult>>,
}

impl BenchmarkSuite {
    /// Create a new benchmark suite
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            config,
            results: RwLock::new(Vec::new()),
        }
    }

    /// Create a benchmark suite with default configuration
    pub fn new_default() -> Self {
        Self::new(BenchmarkConfig::default())
    }

    /// Benchmark a specific operation
    pub fn benchmark_operation<T>(
        &self,
        op_name: &str,
        inputs: &[&Tensor<T>],
        attrs: &HashMap<String, crate::ops::registry::AttrValue>,
    ) -> Result<BenchmarkResult>
    where
        T: Clone + Default + Send + Sync + 'static,
    {
        // Get operation definition
        let op_def = OP_REGISTRY.get_op(op_name).ok_or_else(|| {
            TensorError::invalid_argument(format!("Operation '{op_name}' not found"))
        })?;

        // Determine device and dtype from first input
        let device = *inputs[0].device();
        let dtype = inputs[0].dtype();

        // Get kernel
        let kernel = OP_REGISTRY
            .get_kernel(op_name, device, dtype)
            .ok_or_else(|| {
                TensorError::invalid_argument(format!(
                    "No kernel found for '{op_name}' on {device:?} with {dtype:?}"
                ))
            })?;

        // Prepare inputs for kernel
        let kernel_inputs: Vec<&dyn std::any::Any> =
            inputs.iter().map(|t| *t as &dyn std::any::Any).collect();

        // Warmup phase
        for _ in 0..self.config.warmup_iterations {
            let _ = kernel.compute(&kernel_inputs, attrs)?;
        }

        // Measurement phase
        let mut total_duration = Duration::new(0, 0);
        let mut memory_measurements = Vec::new();

        for _ in 0..self.config.measurement_iterations {
            let memory_before = if self.config.measure_memory {
                Some(self.get_memory_usage(device))
            } else {
                None
            };

            let start = Instant::now();
            let _ = kernel.compute(&kernel_inputs, attrs)?;
            let duration = start.elapsed();

            if duration > self.config.max_execution_time {
                return Err(TensorError::invalid_argument(
                    "Operation took too long to execute".to_string(),
                ));
            }

            total_duration += duration;

            if let Some(mem_before) = memory_before {
                let memory_after = self.get_memory_usage(device);
                memory_measurements.push(memory_after.saturating_sub(mem_before));
            }
        }

        let avg_duration = total_duration / self.config.measurement_iterations as u32;

        // Calculate metrics
        let input_shapes: Vec<Shape> = inputs.iter().map(|t| t.shape().clone()).collect();

        let memory_used = if !memory_measurements.is_empty() {
            Some(memory_measurements.iter().sum::<u64>() / memory_measurements.len() as u64)
        } else {
            None
        };

        let flops = if self.config.calculate_flops {
            self.estimate_flops(op_name, &input_shapes)
        } else {
            None
        };

        let throughput = {
            let total_elements: usize = inputs.iter().map(|t| t.shape().size()).sum();
            Some(total_elements as f64 / avg_duration.as_secs_f64())
        };

        let mut metadata = HashMap::new();
        metadata.insert(
            "warmup_iterations".to_string(),
            self.config.warmup_iterations.to_string(),
        );
        metadata.insert(
            "measurement_iterations".to_string(),
            self.config.measurement_iterations.to_string(),
        );

        let result = BenchmarkResult {
            operation: op_name.to_string(),
            version: op_def.version,
            device,
            dtype,
            input_shapes,
            duration: avg_duration,
            memory_used,
            flops,
            throughput,
            metadata,
        };

        // Store result
        self.results.write().unwrap().push(result.clone());

        Ok(result)
    }

    /// Get all benchmark results
    pub fn results(&self) -> Vec<BenchmarkResult> {
        self.results.read().unwrap().clone()
    }

    /// Clear all benchmark results
    pub fn clear_results(&self) {
        self.results.write().unwrap().clear();
    }

    /// Generate a performance report
    pub fn generate_report(&self) -> String {
        let results = self.results();
        if results.is_empty() {
            return "No benchmark results available.".to_string();
        }

        let mut report = String::new();
        report.push_str("# Kernel Performance Benchmark Report\n\n");

        // Group results by operation
        let mut by_operation: HashMap<String, Vec<&BenchmarkResult>> = HashMap::new();
        for result in &results {
            by_operation
                .entry(result.operation.clone())
                .or_default()
                .push(result);
        }

        for (op_name, op_results) in by_operation {
            report.push_str(&format!("## Operation: {op_name}\n\n"));

            // Sort by performance (fastest first)
            let mut sorted_results = op_results;
            sorted_results.sort_by(|a, b| a.duration.cmp(&b.duration));

            report.push_str(
                "| Device | DType | Duration (μs) | Throughput (elem/s) | Memory (KB) | FLOPS |\n",
            );
            report.push_str(
                "|--------|-------|---------------|-------------------|-------------|-------|\n",
            );

            for result in sorted_results {
                let duration_us = result.duration.as_micros();
                let throughput = result
                    .throughput
                    .map(|t| format!("{t:.2e}"))
                    .unwrap_or_else(|| "N/A".to_string());
                let memory_kb = result
                    .memory_used
                    .map(|m| format!("{:.1}", m as f64 / 1024.0))
                    .unwrap_or_else(|| "N/A".to_string());
                let flops = result
                    .flops
                    .map(|f| format!("{f:.2e}"))
                    .unwrap_or_else(|| "N/A".to_string());

                report.push_str(&format!(
                    "| {:?} | {:?} | {} | {} | {} | {} |\n",
                    result.device, result.dtype, duration_us, throughput, memory_kb, flops
                ));
            }

            report.push('\n');
        }

        report
    }

    /// Export results to JSON
    #[cfg(feature = "serialize")]
    pub fn export_json(&self) -> Result<String> {
        let results = self.results();
        serde_json::to_string_pretty(&results).map_err(|e| TensorError::Other {
            operation: "export_json".to_string(),
            details: format!("JSON serialization failed: {e}"),
            context: None,
        })
    }

    /// Compare current results with baseline results for regression detection
    pub fn detect_regressions(
        &self,
        baseline_results: &[BenchmarkResult],
        threshold: f64,
    ) -> Vec<RegressionReport> {
        let current_results = self.results();
        let mut regressions = Vec::new();

        for current in &current_results {
            // Find matching baseline result
            if let Some(baseline) = baseline_results.iter().find(|b| {
                b.operation == current.operation
                    && b.device == current.device
                    && b.dtype == current.dtype
                    && b.input_shapes == current.input_shapes
            }) {
                let duration_ratio =
                    current.duration.as_secs_f64() / baseline.duration.as_secs_f64();
                let memory_ratio = match (current.memory_used, baseline.memory_used) {
                    (Some(curr_mem), Some(base_mem)) if base_mem > 0 => {
                        Some(curr_mem as f64 / base_mem as f64)
                    }
                    _ => None,
                };

                let is_regression = duration_ratio > (1.0 + threshold)
                    || memory_ratio.is_some_and(|r| r > (1.0 + threshold));

                if is_regression {
                    regressions.push(RegressionReport {
                        operation: current.operation.clone(),
                        device: current.device,
                        dtype: current.dtype,
                        input_shapes: current.input_shapes.clone(),
                        current_duration: current.duration,
                        baseline_duration: baseline.duration,
                        duration_regression: duration_ratio - 1.0,
                        current_memory: current.memory_used,
                        baseline_memory: baseline.memory_used,
                        memory_regression: memory_ratio.map(|r| r - 1.0),
                        severity: if duration_ratio > (1.0 + threshold * 2.0) {
                            RegressionSeverity::High
                        } else {
                            RegressionSeverity::Medium
                        },
                    });
                }
            }
        }

        regressions
    }

    /// Generate statistical summary of benchmark results
    pub fn generate_statistics(&self) -> BenchmarkStatistics {
        let results = self.results();

        if results.is_empty() {
            return BenchmarkStatistics::default();
        }

        let mut duration_stats = StatisticalSummary::new();
        let mut throughput_stats = StatisticalSummary::new();
        let mut memory_stats = StatisticalSummary::new();
        let mut flops_stats = StatisticalSummary::new();

        for result in &results {
            duration_stats.add_sample(result.duration.as_secs_f64());

            if let Some(throughput) = result.throughput {
                throughput_stats.add_sample(throughput);
            }

            if let Some(memory) = result.memory_used {
                memory_stats.add_sample(memory as f64);
            }

            if let Some(flops) = result.flops {
                flops_stats.add_sample(flops);
            }
        }

        BenchmarkStatistics {
            total_benchmarks: results.len(),
            duration_stats: duration_stats.finalize(),
            throughput_stats: throughput_stats.finalize(),
            memory_stats: memory_stats.finalize(),
            flops_stats: flops_stats.finalize(),
        }
    }

    /// Enhanced report generation with statistics
    pub fn generate_detailed_report(&self) -> String {
        let basic_report = self.generate_report();
        let stats = self.generate_statistics();

        let mut report = basic_report;
        report.push_str("\\n## Performance Statistics\\n\\n");
        report.push_str(&format!(
            "**Total Benchmarks:** {}\\n\\n",
            stats.total_benchmarks
        ));

        if let Some(ref duration_stats) = stats.duration_stats {
            report.push_str("**Duration Statistics:**\\n");
            report.push_str(&format!("- Mean: {:.4}s\\n", duration_stats.mean));
            report.push_str(&format!("- Std Dev: {:.4}s\\n", duration_stats.std_dev));
            report.push_str(&format!("- Min: {:.4}s\\n", duration_stats.min));
            report.push_str(&format!("- Max: {:.4}s\\n", duration_stats.max));
            report.push_str(&format!("- Median: {:.4}s\\n\\n", duration_stats.median));
        }

        if let Some(ref throughput_stats) = stats.throughput_stats {
            report.push_str("**Throughput Statistics:**\\n");
            report.push_str(&format!("- Mean: {:.2e} elem/s\\n", throughput_stats.mean));
            report.push_str(&format!(
                "- Std Dev: {:.2e} elem/s\\n",
                throughput_stats.std_dev
            ));
            report.push_str(&format!("- Min: {:.2e} elem/s\\n", throughput_stats.min));
            report.push_str(&format!("- Max: {:.2e} elem/s\\n\\n", throughput_stats.max));
        }

        report
    }

    /// Estimate FLOPS for an operation
    fn estimate_flops(&self, op_name: &str, input_shapes: &[Shape]) -> Option<f64> {
        match op_name {
            // Element-wise operations: 1 FLOP per element
            "Add" | "Sub" | "Mul" | "Div" | "Pow" => {
                input_shapes.first().map(|shape| shape.size() as f64)
            }

            // Activation functions: 1 FLOP per element (simplified)
            "ReLU" | "Sigmoid" | "Tanh" | "LeakyReLU" | "ELU" | "GELU" | "Swish" => {
                input_shapes.first().map(|shape| shape.size() as f64)
            }

            // Trigonometric functions: ~5 FLOPS per element (approximation)
            "Sin" | "Cos" | "Tan" | "Asin" | "Acos" | "Atan" => {
                input_shapes.first().map(|shape| shape.size() as f64 * 5.0)
            }

            // Logarithmic and exponential: ~3 FLOPS per element
            "Log" | "Log2" | "Log10" | "Exp" | "Exp2" | "Sqrt" | "Rsqrt" => {
                input_shapes.first().map(|shape| shape.size() as f64 * 3.0)
            }

            // Special functions: ~8 FLOPS per element (complex calculations)
            "Erf" | "Erfc" | "Gamma" | "LogGamma" | "Bessel" => {
                input_shapes.first().map(|shape| shape.size() as f64 * 8.0)
            }

            "MatMul" => {
                // Matrix multiplication: 2 * M * N * K FLOPS
                if input_shapes.len() >= 2 {
                    let a_shape = &input_shapes[0];
                    let b_shape = &input_shapes[1];
                    if a_shape.rank() >= 2 && b_shape.rank() >= 2 {
                        let a_dims = a_shape.dims();
                        let b_dims = b_shape.dims();
                        let m = a_dims[a_dims.len() - 2] as f64;
                        let k = a_dims[a_dims.len() - 1] as f64;
                        let n = b_dims[b_dims.len() - 1] as f64;

                        // Account for batch dimensions
                        let batch_size = if a_shape.rank() > 2 {
                            a_dims[..a_dims.len() - 2].iter().product::<usize>() as f64
                        } else {
                            1.0
                        };

                        Some(batch_size * 2.0 * m * n * k)
                    } else {
                        None
                    }
                } else {
                    None
                }
            }

            "Conv2D" => {
                // Convolution: more accurate FLOPS estimation
                if input_shapes.len() >= 2 {
                    let input_shape = &input_shapes[0];
                    let filter_shape = &input_shapes[1];

                    if input_shape.rank() == 4 && filter_shape.rank() == 4 {
                        let input_dims = input_shape.dims();
                        let filter_dims = filter_shape.dims();

                        let batch = input_dims[0] as f64;
                        let out_height = input_dims[1] as f64; // Simplified: assume same as input
                        let out_width = input_dims[2] as f64;
                        let out_channels = filter_dims[3] as f64;

                        let filter_height = filter_dims[0] as f64;
                        let filter_width = filter_dims[1] as f64;
                        let in_channels = filter_dims[2] as f64;

                        // FLOPS = batch * out_height * out_width * out_channels * filter_height * filter_width * in_channels * 2
                        Some(
                            batch
                                * out_height
                                * out_width
                                * out_channels
                                * filter_height
                                * filter_width
                                * in_channels
                                * 2.0,
                        )
                    } else {
                        // Fallback to simple estimation
                        input_shapes.first().map(|shape| shape.size() as f64 * 10.0)
                    }
                } else {
                    None
                }
            }

            "BatchNorm" => {
                // Batch normalization: ~6 FLOPS per element (mean, var, normalize, scale, shift)
                input_shapes.first().map(|shape| shape.size() as f64 * 6.0)
            }

            "LayerNorm" => {
                // Layer normalization: ~8 FLOPS per element (more complex than batch norm)
                input_shapes.first().map(|shape| shape.size() as f64 * 8.0)
            }

            "Dropout" => {
                // Dropout: ~1 FLOP per element (just masking)
                input_shapes.first().map(|shape| shape.size() as f64)
            }

            // Element-wise min/max operations: 1 FLOP per element
            "Max" | "Min" => input_shapes.first().map(|shape| shape.size() as f64),

            // Reduction operations: depends on axis, but roughly proportional to input size
            "Sum" | "Mean" | "ArgMax" | "ArgMin" => {
                input_shapes.first().map(|shape| shape.size() as f64)
            }

            "Softmax" => {
                // Softmax: exp + sum + div = ~3 FLOPS per element
                input_shapes.first().map(|shape| shape.size() as f64 * 3.0)
            }

            "CrossEntropy" => {
                // Cross entropy: log + mul + sum = ~3 FLOPS per element
                input_shapes.first().map(|shape| shape.size() as f64 * 3.0)
            }

            // Pooling operations: minimal computation
            "AvgPool" | "MaxPool" => input_shapes.first().map(|shape| shape.size() as f64 * 0.5),

            _ => None, // Unknown operation
        }
    }

    /// Get current memory usage for a device
    fn get_memory_usage(&self, device: Device) -> u64 {
        match device {
            Device::Cpu => {
                // For CPU, use system memory tracking
                #[cfg(target_os = "linux")]
                {
                    self.get_process_memory_linux()
                }
                #[cfg(target_os = "macos")]
                {
                    self.get_process_memory_macos()
                }
                #[cfg(target_os = "windows")]
                {
                    self.get_process_memory_windows()
                }
                #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
                {
                    0 // Fallback for unsupported platforms
                }
            }
            #[cfg(feature = "gpu")]
            Device::Gpu(_) => {
                // For GPU, we could query GPU memory usage via WGPU
                // This requires access to the GPU context
                self.get_gpu_memory_usage()
            }
            #[cfg(feature = "rocm")]
            Device::Rocm(_) => {
                // For ROCM GPU, we could query GPU memory usage
                // This requires access to the ROCM context
                self.get_gpu_memory_usage()
            }
        }
    }

    /// Get process memory usage on Linux by reading /proc/self/status
    #[cfg(target_os = "linux")]
    fn get_process_memory_linux(&self) -> u64 {
        use std::fs;

        if let Ok(status) = fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    if let Some(kb_str) = line.split_whitespace().nth(1) {
                        if let Ok(kb) = kb_str.parse::<u64>() {
                            return kb * 1024; // Convert KB to bytes
                        }
                    }
                }
            }
        }
        0
    }

    /// Get process memory usage on macOS using mach system calls
    #[cfg(target_os = "macos")]
    fn get_process_memory_macos(&self) -> u64 {
        use libc::{c_int, c_uint};
        use std::mem;

        // mach system call constants
        const MACH_TASK_BASIC_INFO: c_int = 20;
        const MACH_TASK_BASIC_INFO_COUNT: c_uint = 10;

        // Declare external mach functions
        extern "C" {
            fn mach_task_self() -> c_uint;
            fn task_info(
                target_task: c_uint,
                flavor: c_int,
                task_info_out: *mut u8,
                task_info_outCnt: *mut c_uint,
            ) -> c_int;
        }

        // Mach task basic info structure
        #[repr(C)]
        struct MachTaskBasicInfo {
            virtual_size: u64,      // virtual memory size (bytes)
            resident_size: u64,     // resident memory size (bytes)
            resident_size_max: u64, // maximum resident memory size (bytes)
            user_time: [u32; 2],    // total user run time
            system_time: [u32; 2],  // total system run time
            policy: c_int,          // default policy for new threads
        }

        unsafe {
            let mut info: MachTaskBasicInfo = mem::zeroed();
            let mut count = MACH_TASK_BASIC_INFO_COUNT;

            let result = task_info(
                mach_task_self(),
                MACH_TASK_BASIC_INFO,
                &mut info as *mut _ as *mut u8,
                &mut count,
            );

            if result == 0 {
                // KERN_SUCCESS
                info.resident_size // Return resident set size in bytes
            } else {
                0 // Error occurred, return 0
            }
        }
    }

    /// Get process memory usage on Windows using Win32 API
    #[cfg(target_os = "windows")]
    fn get_process_memory_windows(&self) -> u64 {
        use std::mem;
        use windows_sys::Win32::Foundation::{BOOL, HANDLE};
        use windows_sys::Win32::System::ProcessStatus::{
            GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
        };

        // Declare external Win32 functions
        extern "system" {
            fn GetCurrentProcess() -> HANDLE;
        }

        unsafe {
            let mut pmc: PROCESS_MEMORY_COUNTERS = mem::zeroed();
            pmc.cb = mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;

            let process_handle = GetCurrentProcess();
            let result: BOOL = GetProcessMemoryInfo(
                process_handle,
                &mut pmc,
                mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
            );

            if result != 0 {
                // Success - return working set size (physical memory currently used)
                pmc.WorkingSetSize as u64
            } else {
                // Error occurred, return 0
                0
            }
        }
    }

    /// Get GPU memory usage
    #[cfg(feature = "gpu")]
    fn get_gpu_memory_usage(&self) -> u64 {
        // This would require access to the WGPU device/adapter
        // to query memory usage. For now, return 0 as a placeholder.
        // In a real implementation, this would need to be passed
        // the GPU context to query memory info.
        0
    }
}

/// Convenience functions for common benchmarking scenarios
/// Benchmark a binary operation (e.g., Add, Mul)
pub fn benchmark_binary_op<T>(
    op_name: &str,
    shape_a: &[usize],
    shape_b: &[usize],
    devices: &[Device],
    _dtypes: &[DType],
) -> Result<Vec<BenchmarkResult>>
where
    T: Clone
        + Default
        + Send
        + Sync
        + 'static
        + num_traits::Zero
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    let mut all_results = Vec::new();

    for &device in devices {
        let suite = BenchmarkSuite::new_default();

        // Create tensors on the target device
        let tensor_a: Tensor<T> = if matches!(device, Device::Cpu) {
            Tensor::zeros(shape_a)
        } else {
            Tensor::zeros(shape_a).to(device)?
        };

        let tensor_b: Tensor<T> = if matches!(device, Device::Cpu) {
            Tensor::zeros(shape_b)
        } else {
            Tensor::zeros(shape_b).to(device)?
        };

        let inputs = vec![&tensor_a, &tensor_b];
        let attrs = HashMap::new();

        match suite.benchmark_operation(op_name, &inputs, &attrs) {
            Ok(result) => all_results.push(result),
            Err(e) => {
                eprintln!("Benchmark failed for {op_name} on {device:?}: {e}");
            }
        }
    }

    Ok(all_results)
}

/// Benchmark a unary operation (e.g., ReLU, Sigmoid)
pub fn benchmark_unary_op<T>(
    op_name: &str,
    input_shape: &[usize],
    devices: &[Device],
    _dtypes: &[DType],
) -> Result<Vec<BenchmarkResult>>
where
    T: Clone
        + Default
        + Send
        + Sync
        + 'static
        + num_traits::Zero
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    let mut all_results = Vec::new();

    for &device in devices {
        let suite = BenchmarkSuite::new_default();

        // Create tensor on the target device
        let tensor: Tensor<T> = if matches!(device, Device::Cpu) {
            Tensor::zeros(input_shape)
        } else {
            Tensor::zeros(input_shape).to(device)?
        };

        let inputs = vec![&tensor];
        let attrs = HashMap::new();

        match suite.benchmark_operation(op_name, &inputs, &attrs) {
            Ok(result) => all_results.push(result),
            Err(e) => {
                eprintln!("Benchmark failed for {op_name} on {device:?}: {e}");
            }
        }
    }

    Ok(all_results)
}

/// Benchmark matrix multiplication with different sizes
pub fn benchmark_matmul_sizes<T>(
    sizes: &[(usize, usize, usize)], // (M, K, N) dimensions
    devices: &[Device],
    _dtypes: &[DType],
) -> Result<Vec<BenchmarkResult>>
where
    T: Clone
        + Default
        + Send
        + Sync
        + 'static
        + num_traits::Zero
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    let mut all_results = Vec::new();

    for &(m, k, n) in sizes {
        for &device in devices {
            let suite = BenchmarkSuite::new_default();

            // Create tensors on the target device
            let tensor_a: Tensor<T> = if matches!(device, Device::Cpu) {
                Tensor::zeros(&[m, k])
            } else {
                Tensor::zeros(&[m, k]).to(device)?
            };

            let tensor_b: Tensor<T> = if matches!(device, Device::Cpu) {
                Tensor::zeros(&[k, n])
            } else {
                Tensor::zeros(&[k, n]).to(device)?
            };

            let inputs = vec![&tensor_a, &tensor_b];
            let attrs = HashMap::new();

            match suite.benchmark_operation("MatMul", &inputs, &attrs) {
                Ok(result) => all_results.push(result),
                Err(e) => {
                    eprintln!("MatMul benchmark failed for {m}x{k}x{n} on {device:?}: {e}");
                }
            }
        }
    }

    Ok(all_results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_config() {
        let config = BenchmarkConfig::default();
        assert_eq!(config.warmup_iterations, 10);
        assert_eq!(config.measurement_iterations, 100);
        assert!(config.measure_memory);
        assert!(config.calculate_flops);
    }

    #[test]
    fn test_benchmark_suite_creation() {
        let suite = BenchmarkSuite::new_default();
        assert!(suite.results().is_empty());
    }

    #[test]
    fn test_flops_estimation() {
        let suite = BenchmarkSuite::new_default();

        // Test Add operation
        let add_flops = suite.estimate_flops("Add", &[Shape::from_slice(&[100, 100])]);
        assert_eq!(add_flops, Some(10000.0));

        // Test MatMul operation
        let matmul_flops = suite.estimate_flops(
            "MatMul",
            &[
                Shape::from_slice(&[64, 128]),
                Shape::from_slice(&[128, 256]),
            ],
        );
        assert_eq!(matmul_flops, Some(2.0 * 64.0 * 256.0 * 128.0));

        // Test unknown operation
        let unknown_flops = suite.estimate_flops("UnknownOp", &[Shape::from_slice(&[100])]);
        assert_eq!(unknown_flops, None);
    }

    #[test]
    fn test_report_generation() {
        let suite = BenchmarkSuite::new_default();
        let empty_report = suite.generate_report();
        assert!(empty_report.contains("No benchmark results available"));
    }

    #[test]
    fn test_enhanced_flops_estimation() {
        let suite = BenchmarkSuite::new_default();

        // Test activation functions
        let relu_flops = suite.estimate_flops("ReLU", &[Shape::from_slice(&[100, 100])]);
        assert_eq!(relu_flops, Some(10000.0));

        // Test trigonometric functions
        let sin_flops = suite.estimate_flops("Sin", &[Shape::from_slice(&[50, 50])]);
        assert_eq!(sin_flops, Some(2500.0 * 5.0));

        // Test special functions
        let erf_flops = suite.estimate_flops("Erf", &[Shape::from_slice(&[10, 10])]);
        assert_eq!(erf_flops, Some(100.0 * 8.0));

        // Test batch MatMul
        let batch_matmul_flops = suite.estimate_flops(
            "MatMul",
            &[
                Shape::from_slice(&[2, 32, 64]),  // batch=2, M=32, K=64
                Shape::from_slice(&[2, 64, 128]), // batch=2, K=64, N=128
            ],
        );
        assert_eq!(batch_matmul_flops, Some(2.0 * 2.0 * 32.0 * 128.0 * 64.0));

        // Test BatchNorm
        let batchnorm_flops = suite.estimate_flops("BatchNorm", &[Shape::from_slice(&[32, 128])]);
        assert_eq!(batchnorm_flops, Some(32.0 * 128.0 * 6.0));
    }

    #[test]
    fn test_statistical_summary() {
        let summary = StatisticalSummary::new();

        // Test empty case
        assert!(summary.finalize().is_none());

        // Test with data
        let mut summary = StatisticalSummary::new();
        summary.add_sample(1.0);
        summary.add_sample(2.0);
        summary.add_sample(3.0);
        summary.add_sample(4.0);
        summary.add_sample(5.0);

        let stats = summary.finalize().unwrap();
        assert_eq!(stats.count, 5);
        assert_eq!(stats.mean, 3.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
        assert!((stats.std_dev - 1.5811).abs() < 0.01); // sqrt(2.5)
    }

    #[test]
    fn test_regression_detection() {
        let suite = BenchmarkSuite::new_default();

        // Create some baseline results
        let baseline_results = vec![BenchmarkResult {
            operation: "Add".to_string(),
            version: crate::ops::registry::OpVersion {
                major: 1,
                minor: 0,
                patch: 0,
            },
            device: Device::Cpu,
            dtype: DType::Float32,
            input_shapes: vec![Shape::from_slice(&[10, 10])],
            duration: Duration::from_millis(1),
            memory_used: Some(1000),
            flops: Some(100.0),
            throughput: Some(10000.0),
            metadata: HashMap::new(),
        }];

        // Add a current result that represents a regression
        let slow_result = BenchmarkResult {
            operation: "Add".to_string(),
            version: crate::ops::registry::OpVersion {
                major: 1,
                minor: 0,
                patch: 0,
            },
            device: Device::Cpu,
            dtype: DType::Float32,
            input_shapes: vec![Shape::from_slice(&[10, 10])],
            duration: Duration::from_millis(3), // 3x slower
            memory_used: Some(2000),            // 2x more memory
            flops: Some(100.0),
            throughput: Some(3333.0), // correspondingly lower
            metadata: HashMap::new(),
        };

        suite.results.write().unwrap().push(slow_result);

        // Detect regressions with 50% threshold
        let regressions = suite.detect_regressions(&baseline_results, 0.5);

        assert_eq!(regressions.len(), 1);
        let regression = &regressions[0];
        assert_eq!(regression.operation, "Add");
        assert_eq!(regression.severity, RegressionSeverity::High); // 200% increase > 100% threshold
        assert!((regression.duration_regression - 2.0).abs() < 0.01); // 3x - 1 = 2 (200% increase)
    }

    #[test]
    fn test_benchmark_statistics_generation() {
        let suite = BenchmarkSuite::new_default();

        // Test empty case
        let empty_stats = suite.generate_statistics();
        assert_eq!(empty_stats.total_benchmarks, 0);
        assert!(empty_stats.duration_stats.is_none());

        // Add some results
        for i in 1..=5 {
            let result = BenchmarkResult {
                operation: format!("Op{i}"),
                version: crate::ops::registry::OpVersion {
                    major: 1,
                    minor: 0,
                    patch: 0,
                },
                device: Device::Cpu,
                dtype: DType::Float32,
                input_shapes: vec![Shape::from_slice(&[10, 10])],
                duration: Duration::from_millis(i),
                memory_used: Some(i * 1000),
                flops: Some(i as f64 * 100.0),
                throughput: Some(i as f64 * 1000.0),
                metadata: HashMap::new(),
            };
            suite.results.write().unwrap().push(result);
        }

        let stats = suite.generate_statistics();
        assert_eq!(stats.total_benchmarks, 5);

        let duration_stats = stats.duration_stats.unwrap();
        assert_eq!(duration_stats.count, 5);
        assert_eq!(duration_stats.mean, 0.003); // 3ms average
        assert_eq!(duration_stats.min, 0.001); // 1ms
        assert_eq!(duration_stats.max, 0.005); // 5ms

        let memory_stats = stats.memory_stats.unwrap();
        assert_eq!(memory_stats.mean, 3000.0); // Average 3000 bytes
    }
}
