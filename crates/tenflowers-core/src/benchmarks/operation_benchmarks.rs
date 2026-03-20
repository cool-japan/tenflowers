use crate::{Device, Result, TensorError};
use crate::benchmarks::{BenchmarkConfig, KernelBenchmark, BenchmarkResult};
use crate::device::async_execution::{AsyncKernel, Priority, kernels::*};
use std::pin::Pin;
use std::future::Future;
use std::time::Duration;
use std::sync::Arc;

/// Comprehensive benchmark suite for GPU manipulation operations
pub struct ManipulationBenchmarks {
    benchmark: Arc<KernelBenchmark>,
}

impl ManipulationBenchmarks {
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            benchmark: Arc::new(KernelBenchmark::new(config)),
        }
    }

    /// Benchmark gather operation with various input sizes
    pub async fn benchmark_gather_operations(&self) -> Result<Vec<BenchmarkResult>> {
        let mut all_results = Vec::new();

        // Test different input sizes
        let test_sizes = vec![
            (1000, 100),    // Small: 1K input, 100 indices
            (10000, 1000),  // Medium: 10K input, 1K indices
            (100000, 10000), // Large: 100K input, 10K indices
            (1000000, 50000), // Very large: 1M input, 50K indices
        ];

        for (input_size, indices_size) in test_sizes {
            let kernel_factory = move |device: Device| {
                TensorManipulationKernel::new_gather(input_size, indices_size)
            };

            let results = self.benchmark.benchmark_kernel(
                &format!("gather_{}x{}", input_size, indices_size),
                kernel_factory
            ).await?;

            all_results.extend(results);
        }

        Ok(all_results)
    }

    /// Benchmark scatter operation with various input sizes
    pub async fn benchmark_scatter_operations(&self) -> Result<Vec<BenchmarkResult>> {
        let mut all_results = Vec::new();

        let test_sizes = vec![
            (1000, 100),
            (10000, 1000),
            (100000, 10000),
            (1000000, 50000),
        ];

        for (tensor_size, updates_size) in test_sizes {
            let kernel_factory = move |device: Device| {
                TensorManipulationKernel::new_scatter(tensor_size, updates_size)
            };

            let results = self.benchmark.benchmark_kernel(
                &format!("scatter_{}x{}", tensor_size, updates_size),
                kernel_factory
            ).await?;

            all_results.extend(results);
        }

        Ok(all_results)
    }

    /// Benchmark roll operation with various tensor sizes
    pub async fn benchmark_roll_operations(&self) -> Result<Vec<BenchmarkResult>> {
        let mut all_results = Vec::new();

        let test_sizes = vec![
            1000,     // Small tensor
            10000,    // Medium tensor
            100000,   // Large tensor
            1000000,  // Very large tensor
        ];

        for size in test_sizes {
            let kernel_factory = move |device: Device| {
                TensorManipulationKernel::new_roll(size)
            };

            let results = self.benchmark.benchmark_kernel(
                &format!("roll_{}", size),
                kernel_factory
            ).await?;

            all_results.extend(results);
        }

        Ok(all_results)
    }

    /// Benchmark where (conditional selection) operation
    pub async fn benchmark_where_operations(&self) -> Result<Vec<BenchmarkResult>> {
        let mut all_results = Vec::new();

        let test_sizes = vec![
            (1000, 3000),     // Small: 1K condition, 3K total
            (10000, 30000),   // Medium: 10K condition, 30K total
            (100000, 300000), // Large: 100K condition, 300K total
            (500000, 1500000), // Very large: 500K condition, 1.5M total
        ];

        for (condition_size, total_size) in test_sizes {
            let kernel_factory = move |device: Device| {
                TensorManipulationKernel::new_where(condition_size, total_size)
            };

            let results = self.benchmark.benchmark_kernel(
                &format!("where_{}x{}", condition_size, total_size),
                kernel_factory
            ).await?;

            all_results.extend(results);
        }

        Ok(all_results)
    }

    /// Run all manipulation operation benchmarks
    pub async fn run_all_benchmarks(&self) -> Result<Vec<BenchmarkResult>> {
        let mut all_results = Vec::new();

        // Run gather benchmarks
        let gather_results = self.benchmark_gather_operations().await?;
        all_results.extend(gather_results);

        // Run scatter benchmarks
        let scatter_results = self.benchmark_scatter_operations().await?;
        all_results.extend(scatter_results);

        // Run roll benchmarks
        let roll_results = self.benchmark_roll_operations().await?;
        all_results.extend(roll_results);

        // Run where benchmarks
        let where_results = self.benchmark_where_operations().await?;
        all_results.extend(where_results);

        Ok(all_results)
    }

    /// Generate a comprehensive report for manipulation operations
    pub async fn generate_manipulation_report(&self) -> Result<String> {
        let results = self.run_all_benchmarks().await?;

        let mut report = String::new();
        report.push_str("# GPU Manipulation Operations Benchmark Report\n\n");

        // Group results by operation type
        let mut operation_groups = std::collections::HashMap::new();
        for result in results {
            let operation = result.kernel_name.split('_').next().unwrap_or("unknown").to_string();
            operation_groups.entry(operation).or_insert_with(Vec::new).push(result);
        }

        for (operation, results) in operation_groups {
            report.push_str(&format!("## {} Operation Performance\n\n", operation.to_uppercase()));
            
            report.push_str("| Input Size | Device | Mean Time | Throughput (ops/s) | Bandwidth (GB/s) |\n");
            report.push_str("|-----------|---------|-----------|-------------------|------------------|\n");

            for result in results {
                let input_size = result.kernel_name.split('_').nth(1).unwrap_or("unknown");
                report.push_str(&format!(
                    "| {} | {:?} | {:?} | {:.2e} | {:.2} |\n",
                    input_size,
                    result.device,
                    result.mean_time,
                    result.throughput_ops_per_sec,
                    result.memory_bandwidth_gb_per_sec
                ));
            }
            
            report.push_str("\n");
        }

        Ok(report)
    }
}

/// Comprehensive benchmark suite for convolution operations
pub struct ConvolutionBenchmarks {
    benchmark: Arc<KernelBenchmark>,
}

impl ConvolutionBenchmarks {
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            benchmark: Arc::new(KernelBenchmark::new(config)),
        }
    }

    /// Benchmark 2D convolution with different sizes
    pub async fn benchmark_conv2d_operations(&self) -> Result<Vec<BenchmarkResult>> {
        let mut all_results = Vec::new();

        // Test different convolution configurations
        let test_configs = vec![
            // (batch_size, in_channels, out_channels, input_size, kernel_size)
            (1, 3, 32, (224, 224), (3, 3)),      // ResNet-like small
            (1, 64, 64, (112, 112), (3, 3)),     // ResNet-like medium
            (1, 128, 256, (56, 56), (3, 3)),     // ResNet-like large
            (8, 3, 64, (224, 224), (7, 7)),      // ResNet first layer
            (8, 64, 128, (112, 112), (1, 1)),    // 1x1 convolution
            (1, 512, 512, (14, 14), (3, 3)),     // Deep network
        ];

        for (batch_size, in_channels, out_channels, input_size, kernel_size) in test_configs {
            let kernel_factory = move |device: Device| {
                ConvKernel {
                    batch_size,
                    in_channels,
                    out_channels,
                    input_size,
                    kernel_size,
                }
            };

            let results = self.benchmark.benchmark_kernel(
                &format!("conv2d_{}x{}x{}_{}x{}_{}x{}", 
                    batch_size, in_channels, out_channels, 
                    input_size.0, input_size.1, 
                    kernel_size.0, kernel_size.1),
                kernel_factory
            ).await?;

            all_results.extend(results);
        }

        Ok(all_results)
    }

    /// Benchmark matrix multiplication operations
    pub async fn benchmark_matmul_operations(&self) -> Result<Vec<BenchmarkResult>> {
        let mut all_results = Vec::new();

        // Test different matrix sizes
        let test_sizes = vec![
            (128, 128, 128, 1),      // Small matrices
            (256, 256, 256, 1),      // Medium matrices
            (512, 512, 512, 1),      // Large matrices
            (1024, 1024, 1024, 1),   // Very large matrices
            (128, 128, 128, 8),      // Small batched
            (256, 256, 256, 8),      // Medium batched
        ];

        for (m, k, n, batch_size) in test_sizes {
            let kernel_factory = move |device: Device| {
                MatMulKernel { m, k, n, batch_size }
            };

            let results = self.benchmark.benchmark_kernel(
                &format!("matmul_{}x{}x{}_batch{}", m, k, n, batch_size),
                kernel_factory
            ).await?;

            all_results.extend(results);
        }

        Ok(all_results)
    }

    /// Benchmark reduction operations
    pub async fn benchmark_reduction_operations(&self) -> Result<Vec<BenchmarkResult>> {
        let mut all_results = Vec::new();

        let test_configs = vec![
            (10000, 1000, 1),     // 1D reduction
            (100000, 10000, 1),   // Large 1D reduction
            (1000000, 1000, 2),   // 2D reduction
            (10000000, 10000, 3), // 3D reduction
        ];

        for (input_size, output_size, num_axes) in test_configs {
            // Test sum reduction
            let sum_kernel_factory = move |device: Device| {
                ReductionKernel::new_sum(input_size, output_size, num_axes)
            };

            let sum_results = self.benchmark.benchmark_kernel(
                &format!("sum_{}_{}_axes{}", input_size, output_size, num_axes),
                sum_kernel_factory
            ).await?;

            all_results.extend(sum_results);

            // Test mean reduction
            let mean_kernel_factory = move |device: Device| {
                ReductionKernel::new_mean(input_size, output_size, num_axes)
            };

            let mean_results = self.benchmark.benchmark_kernel(
                &format!("mean_{}_{}_axes{}", input_size, output_size, num_axes),
                mean_kernel_factory
            ).await?;

            all_results.extend(mean_results);

            // Test max reduction
            let max_kernel_factory = move |device: Device| {
                ReductionKernel::new_max(input_size, output_size, num_axes)
            };

            let max_results = self.benchmark.benchmark_kernel(
                &format!("max_{}_{}_axes{}", input_size, output_size, num_axes),
                max_kernel_factory
            ).await?;

            all_results.extend(max_results);
        }

        Ok(all_results)
    }

    /// Run all convolution-related benchmarks
    pub async fn run_all_benchmarks(&self) -> Result<Vec<BenchmarkResult>> {
        let mut all_results = Vec::new();

        // Run convolution benchmarks
        let conv_results = self.benchmark_conv2d_operations().await?;
        all_results.extend(conv_results);

        // Run matrix multiplication benchmarks
        let matmul_results = self.benchmark_matmul_operations().await?;
        all_results.extend(matmul_results);

        // Run reduction benchmarks
        let reduction_results = self.benchmark_reduction_operations().await?;
        all_results.extend(reduction_results);

        Ok(all_results)
    }

    /// Generate a comprehensive report for convolution operations
    pub async fn generate_convolution_report(&self) -> Result<String> {
        let results = self.run_all_benchmarks().await?;

        let mut report = String::new();
        report.push_str("# Convolution Operations Benchmark Report\n\n");

        // Group results by operation type
        let mut operation_groups = std::collections::HashMap::new();
        for result in results {
            let operation = result.kernel_name.split('_').next().unwrap_or("unknown").to_string();
            operation_groups.entry(operation).or_insert_with(Vec::new).push(result);
        }

        for (operation, results) in operation_groups {
            report.push_str(&format!("## {} Operation Performance\n\n", operation.to_uppercase()));
            
            report.push_str("| Configuration | Device | Mean Time | Throughput (ops/s) | Bandwidth (GB/s) | Efficiency (%) |\n");
            report.push_str("|---------------|---------|-----------|-------------------|------------------|----------------|\n");

            for result in results {
                let config = result.kernel_name.split('_').skip(1).collect::<Vec<_>>().join("_");
                report.push_str(&format!(
                    "| {} | {:?} | {:?} | {:.2e} | {:.2} | {:.1} |\n",
                    config,
                    result.device,
                    result.mean_time,
                    result.throughput_ops_per_sec,
                    result.memory_bandwidth_gb_per_sec,
                    result.efficiency_percentage
                ));
            }
            
            report.push_str("\n");
        }

        Ok(report)
    }
}

/// Comprehensive benchmark runner for all operations
pub struct ComprehensiveBenchmarkSuite {
    manipulation_benchmarks: ManipulationBenchmarks,
    convolution_benchmarks: ConvolutionBenchmarks,
}

impl ComprehensiveBenchmarkSuite {
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            manipulation_benchmarks: ManipulationBenchmarks::new(config.clone()),
            convolution_benchmarks: ConvolutionBenchmarks::new(config),
        }
    }

    /// Run all benchmarks and generate a comprehensive report
    pub async fn run_complete_benchmark_suite(&self) -> Result<String> {
        let mut full_report = String::new();
        
        full_report.push_str("# TenfloweRS Complete Benchmark Suite\n\n");
        full_report.push_str("This report contains comprehensive benchmarks for all recently implemented operations.\n\n");
        
        // Generate manipulation operations report
        let manipulation_report = self.manipulation_benchmarks.generate_manipulation_report().await?;
        full_report.push_str(&manipulation_report);
        full_report.push_str("\n---\n\n");
        
        // Generate convolution operations report
        let convolution_report = self.convolution_benchmarks.generate_convolution_report().await?;
        full_report.push_str(&convolution_report);
        
        Ok(full_report)
    }

    /// Compare CPU vs GPU performance for key operations
    pub async fn generate_cpu_gpu_comparison(&self) -> Result<String> {
        let mut report = String::new();
        report.push_str("# CPU vs GPU Performance Comparison\n\n");
        
        // Run a subset of benchmarks for comparison
        let manipulation_results = self.manipulation_benchmarks.run_all_benchmarks().await?;
        let convolution_results = self.convolution_benchmarks.run_all_benchmarks().await?;
        
        // Group results by operation and compare CPU vs GPU
        let mut all_results = manipulation_results;
        all_results.extend(convolution_results);
        
        let mut operation_comparisons = std::collections::HashMap::new();
        for result in all_results {
            let operation = result.kernel_name.split('_').next().unwrap_or("unknown").to_string();
            operation_comparisons.entry(operation).or_insert_with(Vec::new).push(result);
        }
        
        report.push_str("| Operation | CPU Time | GPU Time | Speedup | Winner |\n");
        report.push_str("|-----------|----------|----------|---------|--------|\n");
        
        for (operation, results) in operation_comparisons {
            let cpu_results: Vec<_> = results.iter().filter(|r| matches!(r.device, Device::Cpu)).collect();
            let gpu_results: Vec<_> = results.iter().filter(|r| !matches!(r.device, Device::Cpu)).collect();
            
            if let (Some(cpu_result), Some(gpu_result)) = (cpu_results.first(), gpu_results.first()) {
                let speedup = cpu_result.mean_time.as_nanos() as f64 / gpu_result.mean_time.as_nanos() as f64;
                let winner = if speedup > 1.0 { "GPU" } else { "CPU" };
                
                report.push_str(&format!(
                    "| {} | {:?} | {:?} | {:.2}x | {} |\n",
                    operation,
                    cpu_result.mean_time,
                    gpu_result.mean_time,
                    speedup.max(1.0 / speedup),
                    winner
                ));
            }
        }
        
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_manipulation_benchmarks_creation() {
        let config = BenchmarkConfig::default();
        let benchmarks = ManipulationBenchmarks::new(config);
        // Test that it creates successfully
        assert!(std::ptr::addr_of!(benchmarks.benchmark).is_null() == false);
    }
    
    #[test]
    fn test_convolution_benchmarks_creation() {
        let config = BenchmarkConfig::default();
        let benchmarks = ConvolutionBenchmarks::new(config);
        // Test that it creates successfully
        assert!(std::ptr::addr_of!(benchmarks.benchmark).is_null() == false);
    }
    
    #[test]
    fn test_comprehensive_suite_creation() {
        let config = BenchmarkConfig::default();
        let suite = ComprehensiveBenchmarkSuite::new(config);
        // Test that it creates successfully - basic smoke test
        assert!(std::mem::size_of_val(&suite) > 0);
    }
}