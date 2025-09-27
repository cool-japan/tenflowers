//! Metal Kernel Performance Benchmarking
//!
//! This module provides comprehensive benchmarking utilities for Metal kernels,
//! allowing performance analysis and optimization of GPU operations.

use super::device::MetalDevice;
use super::types::BenchmarkResult;
#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::{Result, Tensor, TensorError};
use std::time::{Duration, Instant};

/// Metal kernel performance benchmarking suite
#[cfg(all(target_os = "macos", feature = "metal"))]
#[derive(Debug)]
pub struct MetalBenchmark {
    device: MetalDevice,
    results: Vec<BenchmarkResult>,
}

/// Convolution configuration for benchmarking
#[cfg(all(target_os = "macos", feature = "metal"))]
#[derive(Debug, Clone)]
pub struct ConvConfig {
    pub input_shape: Vec<usize>,  // [batch, channels, height, width]
    pub weight_shape: Vec<usize>, // [out_channels, in_channels, kernel_h, kernel_w]
    pub stride: [usize; 2],
    pub padding: [usize; 2],
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl MetalBenchmark {
    /// Create a new benchmark suite
    pub fn new() -> Result<Self> {
        Ok(MetalBenchmark {
            device: MetalDevice::new()?,
            results: Vec::new(),
        })
    }

    /// Benchmark matrix multiplication performance
    pub fn benchmark_matmul(
        &mut self,
        sizes: &[(usize, usize, usize)],
    ) -> Result<Vec<BenchmarkResult>> {
        let mut results = Vec::new();

        for &(m, n, k) in sizes {
            let a = Tensor::<f32>::zeros(&[m, k]);
            let b = Tensor::<f32>::zeros(&[k, n]);

            let start = Instant::now();
            let _result = self.device.matmul_mps(&a, &b)?;
            let duration = start.elapsed();

            let operations = 2 * m * n * k; // FLOPS for matrix multiplication
            let throughput_gops = operations as f64 / duration.as_secs_f64() / 1e9;

            let memory_accessed = (m * k + k * n + m * n) * 4; // bytes for f32
            let memory_bandwidth_gbps = memory_accessed as f64 / duration.as_secs_f64() / 1e9;

            let efficiency_percent =
                self.calculate_efficiency(throughput_gops, memory_bandwidth_gbps);

            results.push(BenchmarkResult {
                operation: format!("matmul_{}x{}x{}", m, n, k),
                config: format!("M={}, N={}, K={}", m, n, k),
                execution_time_ms: duration.as_secs_f64() * 1000.0,
                throughput_gops,
                memory_bandwidth_gbps,
                efficiency_percent,
            });
        }

        self.results.extend(results.clone());
        Ok(results)
    }

    /// Benchmark convolution performance
    pub fn benchmark_conv2d(&mut self, configs: &[ConvConfig]) -> Result<Vec<BenchmarkResult>> {
        let mut results = Vec::new();

        for config in configs {
            let input = Tensor::<f32>::zeros(&config.input_shape);
            let weights = Tensor::<f32>::zeros(&config.weight_shape);

            let start = Instant::now();
            let _result =
                self.device
                    .conv2d_mps(&input, &weights, None, config.stride, config.padding)?;
            let duration = start.elapsed();

            // Estimate FLOPS for convolution
            let output_h = (config.input_shape[2] + 2 * config.padding[0] - config.weight_shape[2])
                / config.stride[0]
                + 1;
            let output_w = (config.input_shape[3] + 2 * config.padding[1] - config.weight_shape[3])
                / config.stride[1]
                + 1;
            let operations = config.input_shape[0]
                * config.weight_shape[0]
                * output_h
                * output_w
                * config.weight_shape[1]
                * config.weight_shape[2]
                * config.weight_shape[3]
                * 2;
            let throughput_gops = operations as f64 / duration.as_secs_f64() / 1e9;

            // Calculate memory bandwidth (simplified)
            let input_bytes = config.input_shape.iter().product::<usize>() * 4;
            let weight_bytes = config.weight_shape.iter().product::<usize>() * 4;
            let output_bytes =
                config.input_shape[0] * config.weight_shape[0] * output_h * output_w * 4;
            let total_bytes = input_bytes + weight_bytes + output_bytes;
            let memory_bandwidth_gbps = total_bytes as f64 / duration.as_secs_f64() / 1e9;

            let efficiency_percent =
                self.calculate_efficiency(throughput_gops, memory_bandwidth_gbps);

            results.push(BenchmarkResult {
                operation: format!("conv2d_{:?}", config.input_shape),
                config: format!(
                    "Input: {:?}, Weight: {:?}, Stride: {:?}, Padding: {:?}",
                    config.input_shape, config.weight_shape, config.stride, config.padding
                ),
                execution_time_ms: duration.as_secs_f64() * 1000.0,
                throughput_gops,
                memory_bandwidth_gbps,
                efficiency_percent,
            });
        }

        self.results.extend(results.clone());
        Ok(results)
    }

    /// Benchmark element-wise operations
    pub fn benchmark_elementwise(&mut self, sizes: &[usize]) -> Result<Vec<BenchmarkResult>> {
        let mut results = Vec::new();

        for &size in sizes {
            let a = Tensor::<f32>::zeros(&[size]);
            let b = Tensor::<f32>::zeros(&[size]);

            // Benchmark addition
            let start = Instant::now();
            let _result =
                self.device
                    .elementwise_coalesced(&a, &b, super::types::ElementwiseOp::Add)?;
            let duration = start.elapsed();

            let operations = size; // One operation per element
            let throughput_gops = operations as f64 / duration.as_secs_f64() / 1e9;

            let memory_accessed = size * 3 * 4; // Read A, read B, write result (f32)
            let memory_bandwidth_gbps = memory_accessed as f64 / duration.as_secs_f64() / 1e9;

            let efficiency_percent =
                self.calculate_efficiency(throughput_gops, memory_bandwidth_gbps);

            results.push(BenchmarkResult {
                operation: format!("elementwise_add_{}", size),
                config: format!("Size: {}", size),
                execution_time_ms: duration.as_secs_f64() * 1000.0,
                throughput_gops,
                memory_bandwidth_gbps,
                efficiency_percent,
            });
        }

        self.results.extend(results.clone());
        Ok(results)
    }

    /// Benchmark reduction operations
    pub fn benchmark_reductions(&mut self, sizes: &[usize]) -> Result<Vec<BenchmarkResult>> {
        let mut results = Vec::new();

        for &size in sizes {
            let tensor = Tensor::<f32>::zeros(&[size]);

            // Benchmark sum reduction
            let start = Instant::now();
            let _result =
                self.device
                    .reduce_optimized(&tensor, super::types::ReductionOp::Sum, None)?;
            let duration = start.elapsed();

            let operations = size; // One addition per element (approximately)
            let throughput_gops = operations as f64 / duration.as_secs_f64() / 1e9;

            let memory_accessed = size * 4; // Read input (f32)
            let memory_bandwidth_gbps = memory_accessed as f64 / duration.as_secs_f64() / 1e9;

            let efficiency_percent =
                self.calculate_efficiency(throughput_gops, memory_bandwidth_gbps);

            results.push(BenchmarkResult {
                operation: format!("reduce_sum_{}", size),
                config: format!("Size: {}", size),
                execution_time_ms: duration.as_secs_f64() * 1000.0,
                throughput_gops,
                memory_bandwidth_gbps,
                efficiency_percent,
            });
        }

        self.results.extend(results.clone());
        Ok(results)
    }

    /// Benchmark activation functions
    pub fn benchmark_activations(&mut self, sizes: &[usize]) -> Result<Vec<BenchmarkResult>> {
        let mut results = Vec::new();
        let activations = [
            super::types::ActivationType::ReLU,
            super::types::ActivationType::GELU,
            super::types::ActivationType::Swish,
            super::types::ActivationType::Tanh,
            super::types::ActivationType::Sigmoid,
        ];

        for &size in sizes {
            let tensor = Tensor::<f32>::zeros(&[size]);

            for activation in &activations {
                let start = Instant::now();
                let _result = self.device.fused_activation(&tensor, *activation)?;
                let duration = start.elapsed();

                let operations = size; // One operation per element
                let throughput_gops = operations as f64 / duration.as_secs_f64() / 1e9;

                let memory_accessed = size * 2 * 4; // Read input, write output (f32)
                let memory_bandwidth_gbps = memory_accessed as f64 / duration.as_secs_f64() / 1e9;

                let efficiency_percent =
                    self.calculate_efficiency(throughput_gops, memory_bandwidth_gbps);

                results.push(BenchmarkResult {
                    operation: format!("activation_{:?}_{}", activation, size),
                    config: format!("Activation: {:?}, Size: {}", activation, size),
                    execution_time_ms: duration.as_secs_f64() * 1000.0,
                    throughput_gops,
                    memory_bandwidth_gbps,
                    efficiency_percent,
                });
            }
        }

        self.results.extend(results.clone());
        Ok(results)
    }

    /// Benchmark memory bandwidth
    pub fn benchmark_memory_bandwidth(&mut self, sizes: &[usize]) -> Result<Vec<BenchmarkResult>> {
        let mut results = Vec::new();

        for &size in sizes {
            let start = Instant::now();
            let (bandwidth, _stats) = self.device.measure_memory_bandwidth::<f32>(size)?;
            let duration = start.elapsed();

            results.push(BenchmarkResult {
                operation: format!("memory_bandwidth_{}", size),
                config: format!("Size: {}", size),
                execution_time_ms: duration.as_secs_f64() * 1000.0,
                throughput_gops: 0.0, // Not applicable for memory bandwidth
                memory_bandwidth_gbps: bandwidth / 1000.0, // Convert MB/s to GB/s
                efficiency_percent: self.calculate_memory_efficiency(bandwidth / 1000.0),
            });
        }

        self.results.extend(results.clone());
        Ok(results)
    }

    /// Run comprehensive benchmark suite
    pub fn run_comprehensive_benchmarks(&mut self) -> Result<Vec<BenchmarkResult>> {
        println!("Running comprehensive Metal kernel benchmarks...");
        println!("{}", "=".repeat(60));

        let mut all_results = Vec::new();

        // Matrix multiplication benchmarks
        println!("Benchmarking matrix multiplication...");
        let matmul_sizes = vec![(256, 256, 256), (512, 512, 512), (1024, 1024, 1024)];
        let matmul_results = self.benchmark_matmul(&matmul_sizes)?;
        all_results.extend(matmul_results);

        // Convolution benchmarks
        println!("Benchmarking convolution operations...");
        let conv_configs = vec![
            ConvConfig {
                input_shape: vec![1, 3, 224, 224],
                weight_shape: vec![64, 3, 7, 7],
                stride: [2, 2],
                padding: [3, 3],
            },
            ConvConfig {
                input_shape: vec![1, 64, 112, 112],
                weight_shape: vec![128, 64, 3, 3],
                stride: [1, 1],
                padding: [1, 1],
            },
        ];
        let conv_results = self.benchmark_conv2d(&conv_configs)?;
        all_results.extend(conv_results);

        // Element-wise operation benchmarks
        println!("Benchmarking element-wise operations...");
        let elementwise_sizes = vec![1024, 4096, 16384, 65536];
        let elementwise_results = self.benchmark_elementwise(&elementwise_sizes)?;
        all_results.extend(elementwise_results);

        // Reduction benchmarks
        println!("Benchmarking reduction operations...");
        let reduction_sizes = vec![1024, 4096, 16384, 65536];
        let reduction_results = self.benchmark_reductions(&reduction_sizes)?;
        all_results.extend(reduction_results);

        // Activation benchmarks
        println!("Benchmarking activation functions...");
        let activation_sizes = vec![1024, 4096, 16384];
        let activation_results = self.benchmark_activations(&activation_sizes)?;
        all_results.extend(activation_results);

        // Memory bandwidth benchmarks
        println!("Benchmarking memory bandwidth...");
        let bandwidth_sizes = vec![1024, 4096, 16384, 65536];
        let bandwidth_results = self.benchmark_memory_bandwidth(&bandwidth_sizes)?;
        all_results.extend(bandwidth_results);

        println!("Benchmark suite completed!");
        Ok(all_results)
    }

    /// Generate detailed performance report
    pub fn generate_report(&self) -> String {
        let mut report = String::from("Metal Kernel Performance Report\n");
        report.push_str(&format!("{}\n\n", "=".repeat(60)));

        // Group results by operation type
        let mut operations_by_type: std::collections::HashMap<String, Vec<&BenchmarkResult>> =
            std::collections::HashMap::new();

        for result in &self.results {
            let op_type = result
                .operation
                .split('_')
                .next()
                .unwrap_or("unknown")
                .to_string();
            operations_by_type
                .entry(op_type)
                .or_insert_with(Vec::new)
                .push(result);
        }

        for (op_type, results) in operations_by_type {
            report.push_str(&format!("## {} Operations\n", op_type.to_uppercase()));
            report.push_str(&format!("{}\n", "-".repeat(40)));

            for result in results {
                report.push_str(&format!("Operation: {}\n", result.operation));
                report.push_str(&format!("  Config: {}\n", result.config));
                report.push_str(&format!(
                    "  Execution Time: {:.2} ms\n",
                    result.execution_time_ms
                ));
                report.push_str(&format!(
                    "  Throughput: {:.2} GOPS\n",
                    result.throughput_gops
                ));
                report.push_str(&format!(
                    "  Memory Bandwidth: {:.2} GB/s\n",
                    result.memory_bandwidth_gbps
                ));
                report.push_str(&format!(
                    "  Efficiency: {:.1}%\n\n",
                    result.efficiency_percent
                ));
            }
        }

        // Summary statistics
        report.push_str("## Summary Statistics\n");
        report.push_str(&format!("{}\n", "-".repeat(40)));

        let total_operations = self.results.len();
        let avg_throughput =
            self.results.iter().map(|r| r.throughput_gops).sum::<f64>() / total_operations as f64;
        let avg_bandwidth = self
            .results
            .iter()
            .map(|r| r.memory_bandwidth_gbps)
            .sum::<f64>()
            / total_operations as f64;
        let avg_efficiency = self
            .results
            .iter()
            .map(|r| r.efficiency_percent)
            .sum::<f64>()
            / total_operations as f64;

        report.push_str(&format!("Total Benchmarks: {}\n", total_operations));
        report.push_str(&format!("Average Throughput: {:.2} GOPS\n", avg_throughput));
        report.push_str(&format!(
            "Average Memory Bandwidth: {:.2} GB/s\n",
            avg_bandwidth
        ));
        report.push_str(&format!("Average Efficiency: {:.1}%\n", avg_efficiency));

        report
    }

    /// Clear benchmark results
    pub fn clear_results(&mut self) {
        self.results.clear();
    }

    /// Get all benchmark results
    pub fn get_results(&self) -> &[BenchmarkResult] {
        &self.results
    }

    // Private helper methods

    fn calculate_efficiency(&self, throughput_gops: f64, memory_bandwidth_gbps: f64) -> f64 {
        // Simplified efficiency calculation based on theoretical peak performance
        // This would be more sophisticated in a real implementation
        let theoretical_peak_gops = 1000.0; // Placeholder value
        let theoretical_peak_bandwidth = 400.0; // GB/s for high-end Apple Silicon

        let compute_efficiency = (throughput_gops / theoretical_peak_gops) * 100.0;
        let memory_efficiency = (memory_bandwidth_gbps / theoretical_peak_bandwidth) * 100.0;

        // Return the lower of compute and memory efficiency (bottleneck)
        compute_efficiency.min(memory_efficiency).min(100.0)
    }

    fn calculate_memory_efficiency(&self, bandwidth_gbps: f64) -> f64 {
        let theoretical_peak_bandwidth = 400.0; // GB/s for high-end Apple Silicon
        ((bandwidth_gbps / theoretical_peak_bandwidth) * 100.0).min(100.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn test_benchmark_creation() {
        let result = MetalBenchmark::new();
        // Test should pass on macOS with Metal support
        assert!(result.is_ok() || result.unwrap_err().to_string().contains("No Metal device"));
    }

    #[test]
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn test_conv_config_creation() {
        let config = ConvConfig {
            input_shape: vec![1, 3, 224, 224],
            weight_shape: vec![64, 3, 7, 7],
            stride: [2, 2],
            padding: [3, 3],
        };

        assert_eq!(config.input_shape, vec![1, 3, 224, 224]);
        assert_eq!(config.weight_shape, vec![64, 3, 7, 7]);
        assert_eq!(config.stride, [2, 2]);
        assert_eq!(config.padding, [3, 3]);
    }

    #[test]
    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    fn test_benchmarks_not_available() {
        // On non-macOS platforms, Metal benchmarks are not available
        // This test ensures the module compiles correctly on all platforms
        assert!(true);
    }
}
