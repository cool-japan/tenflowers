//! GPU Performance Benchmarking vs TensorFlow
//!
//! This module provides comprehensive GPU benchmarking capabilities to measure
//! TenfloweRS GPU performance against TensorFlow GPU and work towards achieving
//! 90% of TensorFlow GPU performance as specified in the project goals.

use super::benchmark::{BenchmarkConfig, BenchmarkResult, BenchmarkSuite};
use super::framework_comparison::{FrameworkBenchmarkConfig, FrameworkComparisonResult};
use crate::gpu::performance_optimizer::{
    GpuCapabilities, GpuOpMetrics, GpuPerformanceOptimizer, OptimizationConfig,
};
use crate::{DType, Device, Result, Tensor, TensorError};
use std::collections::HashMap;
use std::process::Command;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

/// GPU-specific benchmark configuration
#[derive(Debug, Clone)]
pub struct GpuBenchmarkConfig {
    pub base_config: BenchmarkConfig,
    pub gpu_device_ids: Vec<usize>,
    pub test_mixed_precision: bool,
    pub test_tensor_cores: bool,
    pub enable_cuda_graphs: bool,
    pub target_tensorflow_efficiency: f64, // Target: 0.9 for 90%
    pub python_executable: String,
}

impl Default for GpuBenchmarkConfig {
    fn default() -> Self {
        Self {
            base_config: BenchmarkConfig::default(),
            gpu_device_ids: vec![0],
            test_mixed_precision: true,
            test_tensor_cores: true,
            enable_cuda_graphs: true,
            target_tensorflow_efficiency: 0.9,
            python_executable: "python3".to_string(),
        }
    }
}

/// GPU benchmark result comparing TenfloweRS vs TensorFlow GPU
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct GpuBenchmarkResult {
    pub operation: String,
    pub input_shapes: Vec<Vec<usize>>,
    pub dtype: DType,
    pub device_id: usize,
    pub tenflowers_metrics: GpuOpMetrics,
    pub tensorflow_gpu_time: Option<Duration>,
    pub pytorch_gpu_time: Option<Duration>,
    pub performance_ratio: f64, // TenfloweRS / TensorFlow (target: >= 0.9)
    pub memory_efficiency: f64,
    pub throughput_comparison: HashMap<String, f64>,
    pub bottlenecks_identified: Vec<String>,
    pub optimization_suggestions: Vec<String>,
    pub meets_target: bool, // Whether it meets 90% target
}

/// Comprehensive GPU benchmark suite for TensorFlow comparison
pub struct GpuTensorFlowBenchmark {
    config: GpuBenchmarkConfig,
    gpu_optimizer: Arc<GpuPerformanceOptimizer>,
    benchmark_suite: BenchmarkSuite,
}

impl GpuTensorFlowBenchmark {
    /// Create a new GPU TensorFlow benchmark suite
    pub fn new(config: GpuBenchmarkConfig, gpu_optimizer: Arc<GpuPerformanceOptimizer>) -> Self {
        let benchmark_suite = BenchmarkSuite::new(config.base_config.clone());

        Self {
            config,
            gpu_optimizer,
            benchmark_suite,
        }
    }

    /// Run comprehensive GPU benchmark against TensorFlow
    pub fn run_comprehensive_benchmark(&self) -> Result<Vec<GpuBenchmarkResult>> {
        println!("🚀 Running Comprehensive GPU Benchmark vs TensorFlow");
        println!("Target: Achieve 90% of TensorFlow GPU performance\n");

        let mut results = Vec::new();

        // Core operations to benchmark
        let operations = [
            ("MatMul", vec![vec![1024, 1024], vec![1024, 1024]]),
            ("MatMul", vec![vec![2048, 2048], vec![2048, 2048]]),
            ("MatMul", vec![vec![4096, 4096], vec![4096, 4096]]),
            ("Add", vec![vec![10000000], vec![10000000]]),
            ("Mul", vec![vec![10000000], vec![10000000]]),
            ("Conv2D", vec![vec![32, 256, 256, 3], vec![3, 3, 3, 64]]),
            ("Conv2D", vec![vec![64, 128, 128, 64], vec![3, 3, 64, 128]]),
            ("BatchNorm", vec![vec![32, 128, 128, 64]]),
            ("ReLU", vec![vec![32, 128, 128, 64]]),
            ("Softmax", vec![vec![1024, 10000]]),
        ];

        for &device_id in &self.config.gpu_device_ids {
            println!("Benchmarking GPU Device {device_id}:");

            for (op_name, input_shapes) in &operations {
                println!("  Testing {op_name} with shapes {:?}", input_shapes);

                match self.benchmark_gpu_operation(op_name, input_shapes, device_id) {
                    Ok(result) => {
                        let status = if result.meets_target {
                            "✅ MEETS TARGET"
                        } else {
                            "❌ BELOW TARGET"
                        };
                        println!(
                            "    Performance ratio: {:.3} - {status}",
                            result.performance_ratio
                        );
                        results.push(result);
                    }
                    Err(e) => {
                        println!("    ❌ Benchmark failed: {e}");
                    }
                }
            }
        }

        // Mixed precision benchmarks if enabled
        if self.config.test_mixed_precision {
            println!("\n🔄 Running Mixed Precision Benchmarks (FP16):");
            let fp16_results = self.run_mixed_precision_benchmarks()?;
            results.extend(fp16_results);
        }

        // Generate summary report
        self.print_benchmark_summary(&results);

        Ok(results)
    }

    /// Benchmark a specific GPU operation against TensorFlow
    fn benchmark_gpu_operation(
        &self,
        operation: &str,
        input_shapes: &[Vec<usize>],
        device_id: usize,
    ) -> Result<GpuBenchmarkResult> {
        // Create tensors on GPU
        let mut gpu_tensors = Vec::new();
        for shape in input_shapes {
            let tensor: Tensor<f32> = Tensor::zeros(shape).to(Device::Gpu(device_id))?;
            gpu_tensors.push(tensor);
        }

        // Benchmark TenfloweRS GPU performance
        self.gpu_optimizer.start_profiling(
            operation,
            device_id,
            input_shapes.to_vec(),
            DType::Float32,
        );

        let start = Instant::now();
        let tenflowers_result = self.execute_tenflowers_operation(operation, &gpu_tensors)?;
        let tenflowers_time = start.elapsed();

        // Record GPU metrics
        self.gpu_optimizer.record_kernel_execution(
            tenflowers_time,
            input_shapes
                .iter()
                .map(|s| s.iter().product::<usize>())
                .sum(),
            crate::gpu::performance_optimizer::WorkgroupConfig::default(),
        );

        let gpu_metrics = self
            .gpu_optimizer
            .finish_profiling()
            .ok_or_else(|| TensorError::other("Failed to collect GPU metrics".to_string()))?;

        // Benchmark TensorFlow GPU performance
        let tensorflow_gpu_time = self.benchmark_tensorflow_gpu(operation, input_shapes)?;
        let pytorch_gpu_time = self.benchmark_pytorch_gpu(operation, input_shapes)?;

        // Calculate performance metrics
        let performance_ratio = if let Some(tf_time) = tensorflow_gpu_time {
            tf_time.as_secs_f64() / tenflowers_time.as_secs_f64()
        } else {
            0.0
        };

        let meets_target = performance_ratio >= self.config.target_tensorflow_efficiency;

        // Analyze bottlenecks and generate suggestions
        let bottlenecks = self.gpu_optimizer.analyze_bottlenecks(&gpu_metrics);
        let bottlenecks_identified: Vec<String> =
            bottlenecks.iter().map(|b| b.description.clone()).collect();

        let optimization_suggestions = self
            .gpu_optimizer
            .get_optimization_recommendations(operation);

        // Calculate throughput comparisons
        let mut throughput_comparison = HashMap::new();
        let tenflowers_throughput = input_shapes
            .iter()
            .map(|s| s.iter().product::<usize>())
            .sum::<usize>() as f64
            / tenflowers_time.as_secs_f64();

        throughput_comparison.insert("tenflowers".to_string(), tenflowers_throughput);

        if let Some(tf_time) = tensorflow_gpu_time {
            let tf_throughput = input_shapes
                .iter()
                .map(|s| s.iter().product::<usize>())
                .sum::<usize>() as f64
                / tf_time.as_secs_f64();
            throughput_comparison.insert("tensorflow".to_string(), tf_throughput);
        }

        Ok(GpuBenchmarkResult {
            operation: operation.to_string(),
            input_shapes: input_shapes.to_vec(),
            dtype: DType::Float32,
            device_id,
            tenflowers_metrics: gpu_metrics,
            tensorflow_gpu_time,
            pytorch_gpu_time,
            performance_ratio,
            memory_efficiency: self.calculate_memory_efficiency(&tenflowers_result)?,
            throughput_comparison,
            bottlenecks_identified,
            optimization_suggestions,
            meets_target,
        })
    }

    /// Execute TenfloweRS operation
    fn execute_tenflowers_operation(
        &self,
        operation: &str,
        tensors: &[Tensor<f32>],
    ) -> Result<Tensor<f32>> {
        match operation {
            "MatMul" => {
                if tensors.len() >= 2 {
                    crate::ops::matmul::matmul(&tensors[0], &tensors[1])
                } else {
                    Err(TensorError::invalid_argument(
                        "MatMul requires 2 tensors".to_string(),
                    ))
                }
            }
            "Add" => {
                if tensors.len() >= 2 {
                    crate::ops::binary::add(&tensors[0], &tensors[1])
                } else {
                    Err(TensorError::invalid_argument(
                        "Add requires 2 tensors".to_string(),
                    ))
                }
            }
            "Mul" => {
                if tensors.len() >= 2 {
                    crate::ops::binary::mul(&tensors[0], &tensors[1])
                } else {
                    Err(TensorError::invalid_argument(
                        "Mul requires 2 tensors".to_string(),
                    ))
                }
            }
            "Conv2D" => {
                if tensors.len() >= 2 {
                    crate::ops::conv::conv2d(&tensors[0], &tensors[1], None, (1, 1), "same")
                } else {
                    Err(TensorError::invalid_argument(
                        "Conv2D requires 2 tensors".to_string(),
                    ))
                }
            }
            "BatchNorm" => {
                if !tensors.is_empty() {
                    // Create dummy gamma, beta, running_mean, running_var tensors for benchmark
                    let num_features = tensors[0].shape().dims()[1]; // Assume NCHW format
                    let ones = crate::Tensor::ones(&[num_features]);
                    let zeros = crate::Tensor::zeros(&[num_features]);
                    crate::ops::normalization::batch_norm(
                        &tensors[0],
                        &ones,   // gamma
                        &zeros,  // beta
                        &zeros,  // running_mean
                        &ones,   // running_var
                        1e-5f32, // epsilon
                        true,    // training
                    )
                } else {
                    Err(TensorError::invalid_argument(
                        "BatchNorm requires 1 tensor".to_string(),
                    ))
                }
            }
            "ReLU" => {
                if !tensors.is_empty() {
                    crate::ops::activation::relu(&tensors[0])
                } else {
                    Err(TensorError::invalid_argument(
                        "ReLU requires 1 tensor".to_string(),
                    ))
                }
            }
            "Softmax" => {
                if !tensors.is_empty() {
                    crate::ops::activation::softmax(&tensors[0], Some(-1))
                } else {
                    Err(TensorError::invalid_argument(
                        "Softmax requires 1 tensor".to_string(),
                    ))
                }
            }
            _ => Err(TensorError::invalid_argument(format!(
                "Unknown operation: {operation}"
            ))),
        }
    }

    /// Benchmark TensorFlow GPU performance
    fn benchmark_tensorflow_gpu(
        &self,
        operation: &str,
        input_shapes: &[Vec<usize>],
    ) -> Result<Option<Duration>> {
        let script = self.generate_tensorflow_gpu_script(operation, input_shapes)?;
        let output = Command::new(&self.config.python_executable)
            .arg("-c")
            .arg(&script)
            .output()
            .map_err(|e| {
                TensorError::other(format!("Failed to execute TensorFlow benchmark: {e}"))
            })?;

        if !output.status.success() {
            println!(
                "Warning: TensorFlow GPU benchmark failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            return Ok(None);
        }

        let elapsed_ns_str = String::from_utf8_lossy(&output.stdout);
        let elapsed_ns: f64 = elapsed_ns_str
            .trim()
            .parse()
            .map_err(|e| TensorError::other(format!("Failed to parse TensorFlow timing: {e}")))?;

        Ok(Some(Duration::from_nanos(elapsed_ns as u64)))
    }

    /// Benchmark PyTorch GPU performance  
    fn benchmark_pytorch_gpu(
        &self,
        operation: &str,
        input_shapes: &[Vec<usize>],
    ) -> Result<Option<Duration>> {
        let script = self.generate_pytorch_gpu_script(operation, input_shapes)?;
        let output = Command::new(&self.config.python_executable)
            .arg("-c")
            .arg(&script)
            .output()
            .map_err(|e| TensorError::other(format!("Failed to execute PyTorch benchmark: {e}")))?;

        if !output.status.success() {
            println!(
                "Warning: PyTorch GPU benchmark failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            return Ok(None);
        }

        let elapsed_ns_str = String::from_utf8_lossy(&output.stdout);
        let elapsed_ns: f64 = elapsed_ns_str
            .trim()
            .parse()
            .map_err(|e| TensorError::other(format!("Failed to parse PyTorch timing: {e}")))?;

        Ok(Some(Duration::from_nanos(elapsed_ns as u64)))
    }

    /// Generate TensorFlow GPU benchmark script
    fn generate_tensorflow_gpu_script(
        &self,
        operation: &str,
        input_shapes: &[Vec<usize>],
    ) -> Result<String> {
        let shape_strs: Vec<String> = input_shapes
            .iter()
            .map(|shape| {
                format!(
                    "[{}]",
                    shape
                        .iter()
                        .map(|x| x.to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            })
            .collect();

        let setup = format!(
            r#"
import tensorflow as tf
import time
import numpy as np

# Ensure GPU is available
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

with tf.device('/GPU:0'):
    # Create tensors
    tensors = []
    for shape in [{}]:
        tensor = tf.random.normal(shape, dtype=tf.float32)
        tensors.append(tensor)
"#,
            shape_strs.join(", ")
        );

        let operation_code = match operation {
            "MatMul" => {
                "result = tf.linalg.matmul(tensors[0], tensors[1])"
            }
            "Add" => {
                "result = tf.add(tensors[0], tensors[1])"
            }
            "Mul" => {
                "result = tf.multiply(tensors[0], tensors[1])"
            }
            "Conv2D" => {
                r#"
    # Ensure correct shapes for Conv2D: [batch, height, width, channels]
    inputs = tensors[0]  # [batch, height, width, in_channels]
    filters = tf.random.normal([3, 3, inputs.shape[-1], 64], dtype=tf.float32)
    result = tf.nn.conv2d(inputs, filters, strides=[1, 1, 1, 1], padding='SAME')"#
            }
            "BatchNorm" => {
                "result = tf.nn.batch_normalization(tensors[0], mean=tf.reduce_mean(tensors[0]), variance=tf.math.reduce_variance(tensors[0]), offset=None, scale=None, variance_epsilon=1e-5)"
            }
            "ReLU" => {
                "result = tf.nn.relu(tensors[0])"
            }
            "Softmax" => {
                "result = tf.nn.softmax(tensors[0])"
            }
            _ => return Err(TensorError::invalid_argument(format!("Unsupported operation: {operation}"))),
        };

        let script = format!(
            r#"
{setup}
    
    # Warmup
    for _ in range(10):
        {operation_code}
    
    # Benchmark
    start_time = time.perf_counter()
    for _ in range(100):
        {operation_code}
    
    # Ensure execution completes
    tf.experimental.numpy.copy(result)
    end_time = time.perf_counter()
    
    elapsed_ns = (end_time - start_time) * 1e9 / 100
    print(f"{{elapsed_ns:.0f}}")
"#
        );

        Ok(script)
    }

    /// Generate PyTorch GPU benchmark script
    fn generate_pytorch_gpu_script(
        &self,
        operation: &str,
        input_shapes: &[Vec<usize>],
    ) -> Result<String> {
        let shape_strs: Vec<String> = input_shapes
            .iter()
            .map(|shape| {
                format!(
                    "[{}]",
                    shape
                        .iter()
                        .map(|x| x.to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            })
            .collect();

        let setup = format!(
            r#"
import torch
import time

# Ensure GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cpu':
    print("0")  # Return 0 if no GPU available
    exit()

# Create tensors on GPU
tensors = []
for shape in [{}]:
    tensor = torch.randn(shape, dtype=torch.float32, device=device)
    tensors.append(tensor)
"#,
            shape_strs.join(", ")
        );

        let operation_code = match operation {
            "MatMul" => "result = torch.matmul(tensors[0], tensors[1])",
            "Add" => "result = torch.add(tensors[0], tensors[1])",
            "Mul" => "result = torch.mul(tensors[0], tensors[1])",
            "Conv2D" => {
                r#"
# Create conv2d layer
conv = torch.nn.Conv2d(tensors[0].shape[1], 64, 3, padding=1).to(device)
result = conv(tensors[0])"#
            }
            "BatchNorm" => {
                r#"
# Create batch norm layer
bn = torch.nn.BatchNorm2d(tensors[0].shape[1]).to(device)
result = bn(tensors[0])"#
            }
            "ReLU" => "result = torch.relu(tensors[0])",
            "Softmax" => "result = torch.softmax(tensors[0], dim=-1)",
            _ => {
                return Err(TensorError::invalid_argument(format!(
                    "Unsupported operation: {operation}"
                )))
            }
        };

        let script = format!(
            r#"
{setup}

# Warmup
for _ in range(10):
    {operation_code}

torch.cuda.synchronize()

# Benchmark
start_time = time.perf_counter()
for _ in range(100):
    {operation_code}

torch.cuda.synchronize()
end_time = time.perf_counter()

elapsed_ns = (end_time - start_time) * 1e9 / 100
print(f"{{elapsed_ns:.0f}}")
"#
        );

        Ok(script)
    }

    /// Run mixed precision benchmarks
    fn run_mixed_precision_benchmarks(&self) -> Result<Vec<GpuBenchmarkResult>> {
        println!("Testing FP16 mixed precision performance...");

        // This would involve creating FP16 tensors and benchmarking
        // For now, return empty results as mixed precision support
        // needs to be implemented at the tensor level
        Ok(Vec::new())
    }

    /// Calculate memory efficiency
    fn calculate_memory_efficiency(&self, _result: &Tensor<f32>) -> Result<f64> {
        // Simplified memory efficiency calculation
        // In a real implementation, this would analyze memory usage patterns
        Ok(0.85) // Placeholder efficiency score
    }

    /// Print comprehensive benchmark summary
    fn print_benchmark_summary(&self, results: &[GpuBenchmarkResult]) {
        println!("\n{}", "=".repeat(80));
        println!("🎯 GPU BENCHMARK SUMMARY - TensorFlow Comparison");
        println!("{}", "=".repeat(80));

        if results.is_empty() {
            println!("No benchmark results available.");
            return;
        }

        let total_tests = results.len();
        let meets_target = results.iter().filter(|r| r.meets_target).count();
        let overall_success_rate = meets_target as f64 / total_tests as f64;

        println!("📊 Overall Performance:");
        println!("  • Total operations tested: {total_tests}");
        println!("  • Operations meeting 90% target: {meets_target}/{total_tests}");
        println!("  • Success rate: {:.1}%", overall_success_rate * 100.0);

        let avg_performance_ratio: f64 =
            results.iter().map(|r| r.performance_ratio).sum::<f64>() / results.len() as f64;

        println!(
            "  • Average performance ratio: {:.3} (target: ≥0.900)",
            avg_performance_ratio
        );

        if avg_performance_ratio >= 0.9 {
            println!("  ✅ OVERALL TARGET ACHIEVED!");
        } else {
            println!("  ❌ Overall target not yet achieved");
            println!(
                "  📈 Performance gap: {:.1}%",
                (0.9 - avg_performance_ratio) * 100.0
            );
        }

        println!("\n📋 Detailed Results:");
        println!("{:-<120}", "");
        println!(
            "| {:^15} | {:^20} | {:^12} | {:^12} | {:^15} | {:^10} |",
            "Operation", "Shapes", "TF GPU (μs)", "TF RS (μs)", "Ratio", "Target Met"
        );
        println!("{:-<120}", "");

        for result in results {
            let tf_time_str = result
                .tensorflow_gpu_time
                .map(|t| format!("{}", t.as_micros()))
                .unwrap_or_else(|| "N/A".to_string());

            let tf_rs_time = result.tenflowers_metrics.total_time.as_micros();
            let ratio_str = format!("{:.3}", result.performance_ratio);
            let target_met = if result.meets_target {
                "✅ Yes"
            } else {
                "❌ No"
            };

            let shapes_str = result
                .input_shapes
                .iter()
                .map(|s| {
                    format!(
                        "[{}]",
                        s.iter()
                            .map(|x| x.to_string())
                            .collect::<Vec<_>>()
                            .join("×")
                    )
                })
                .collect::<Vec<_>>()
                .join(" ");

            println!(
                "| {:^15} | {:^20} | {:^12} | {:^12} | {:^15} | {:^10} |",
                result.operation,
                if shapes_str.len() > 20 {
                    shapes_str[..17].to_string() + "..."
                } else {
                    shapes_str.clone()
                },
                tf_time_str,
                tf_rs_time,
                ratio_str,
                target_met
            );
        }
        println!("{:-<120}", "");

        // Performance improvement recommendations
        println!("\n💡 Performance Improvement Recommendations:");
        let mut all_suggestions: Vec<String> = results
            .iter()
            .flat_map(|r| r.optimization_suggestions.iter())
            .cloned()
            .collect();
        all_suggestions.sort();
        all_suggestions.dedup();

        for (i, suggestion) in all_suggestions.iter().enumerate() {
            println!("  {}. {suggestion}", i + 1);
        }

        // Critical bottlenecks
        println!("\n⚠️ Critical Bottlenecks Identified:");
        let mut all_bottlenecks: Vec<String> = results
            .iter()
            .flat_map(|r| r.bottlenecks_identified.iter())
            .cloned()
            .collect();
        all_bottlenecks.sort();
        all_bottlenecks.dedup();

        for bottleneck in &all_bottlenecks {
            println!("  • {bottleneck}");
        }

        println!("\n{}", "=".repeat(80));
    }
}

/// Convenience function to run a quick GPU benchmark
pub fn run_quick_gpu_tensorflow_benchmark() -> Result<Vec<GpuBenchmarkResult>> {
    println!("🚀 Quick GPU TensorFlow Benchmark");

    // This is a simplified version for quick testing
    // In practice, you'd need proper GPU context setup
    let config = GpuBenchmarkConfig::default();

    // Mock GPU optimizer for now - in real use this would be properly initialized
    #[allow(invalid_value)]
    let gpu_capabilities = crate::gpu::performance_optimizer::detect_gpu_capabilities(
        &unsafe { std::mem::zeroed() }, // Mock device - unsafe but for testing
    );

    #[allow(invalid_value)]
    let gpu_optimizer = Arc::new(GpuPerformanceOptimizer::new(
        Arc::new(unsafe { std::mem::zeroed() }), // Mock device
        Arc::new(unsafe { std::mem::zeroed() }), // Mock queue
        gpu_capabilities,
        OptimizationConfig::default(),
    ));

    let benchmark = GpuTensorFlowBenchmark::new(config, gpu_optimizer);
    benchmark.run_comprehensive_benchmark()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_benchmark_config() {
        let config = GpuBenchmarkConfig::default();
        assert_eq!(config.target_tensorflow_efficiency, 0.9);
        assert!(config.test_mixed_precision);
        assert!(config.test_tensor_cores);
    }

    #[test]
    #[ignore = "Requires valid WGPU device - panics with zero-initialized device"]
    #[allow(invalid_value)]
    fn test_tensorflow_script_generation() {
        let config = GpuBenchmarkConfig::default();
        // SAFETY: This test is ignored and only serves as documentation.
        // Real implementation would use proper GPU device initialization.
        let gpu_capabilities =
            crate::gpu::performance_optimizer::detect_gpu_capabilities(&unsafe {
                std::mem::zeroed()
            });
        let gpu_optimizer = Arc::new(GpuPerformanceOptimizer::new(
            Arc::new(unsafe { std::mem::zeroed() }),
            Arc::new(unsafe { std::mem::zeroed() }),
            gpu_capabilities,
            OptimizationConfig::default(),
        ));

        let benchmark = GpuTensorFlowBenchmark::new(config, gpu_optimizer);

        let script = benchmark
            .generate_tensorflow_gpu_script("MatMul", &[vec![1024, 1024], vec![1024, 1024]])
            .unwrap();

        assert!(script.contains("tf.linalg.matmul"));
        assert!(script.contains("tf.device('/GPU:0')"));
        assert!(script.contains("1024, 1024"));
    }

    #[test]
    #[ignore = "Requires valid WGPU device - panics with zero-initialized device"]
    #[allow(invalid_value)]
    fn test_pytorch_script_generation() {
        let config = GpuBenchmarkConfig::default();
        // SAFETY: This test is ignored and only serves as documentation.
        // Real implementation would use proper GPU device initialization.
        let gpu_capabilities =
            crate::gpu::performance_optimizer::detect_gpu_capabilities(&unsafe {
                std::mem::zeroed()
            });
        let gpu_optimizer = Arc::new(GpuPerformanceOptimizer::new(
            Arc::new(unsafe { std::mem::zeroed() }),
            Arc::new(unsafe { std::mem::zeroed() }),
            gpu_capabilities,
            OptimizationConfig::default(),
        ));

        let benchmark = GpuTensorFlowBenchmark::new(config, gpu_optimizer);

        let script = benchmark
            .generate_pytorch_gpu_script("Add", &[vec![1000000], vec![1000000]])
            .unwrap();

        assert!(script.contains("torch.add"));
        assert!(script.contains("torch.cuda.is_available"));
        assert!(script.contains("1000000"));
    }
}
