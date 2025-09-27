//! Framework comparison benchmarking
//!
//! This module provides utilities to benchmark TenfloweRS against other
//! machine learning frameworks like PyTorch, TensorFlow, and NumPy.

use super::performance_benchmark::BenchmarkConfig;
use crate::{Result, Tensor, TensorError};
use std::collections::HashMap;
use std::process::Command;
use std::time::{Duration, Instant};

/// Framework comparison result
#[derive(Debug, Clone)]
pub struct FrameworkComparisonResult {
    pub operation: String,
    pub size: usize,
    pub tenflowers_time: Duration,
    pub framework_times: HashMap<String, Duration>,
    pub tenflowers_throughput: f64,
    pub framework_throughputs: HashMap<String, f64>,
    pub relative_performance: HashMap<String, f64>, // TenfloweRS vs other frameworks
}

impl FrameworkComparisonResult {
    pub fn new(
        operation: String,
        size: usize,
        tenflowers_time: Duration,
        framework_times: HashMap<String, Duration>,
    ) -> Self {
        let tenflowers_throughput = size as f64 / tenflowers_time.as_secs_f64();

        let mut framework_throughputs = HashMap::new();
        let mut relative_performance = HashMap::new();

        for (framework, time) in &framework_times {
            let throughput = size as f64 / time.as_secs_f64();
            framework_throughputs.insert(framework.clone(), throughput);

            // Relative performance: TenfloweRS time / framework time
            // > 1.0 means TenfloweRS is slower, < 1.0 means TenfloweRS is faster
            let relative = tenflowers_time.as_nanos() as f64 / time.as_nanos() as f64;
            relative_performance.insert(framework.clone(), relative);
        }

        Self {
            operation,
            size,
            tenflowers_time,
            framework_times,
            tenflowers_throughput,
            framework_throughputs,
            relative_performance,
        }
    }
}

/// Framework benchmarking configuration
#[derive(Debug, Clone)]
pub struct FrameworkBenchmarkConfig {
    pub base_config: BenchmarkConfig,
    pub frameworks_to_test: Vec<String>,
    pub python_executable: String,
    pub skip_missing_frameworks: bool,
}

impl Default for FrameworkBenchmarkConfig {
    fn default() -> Self {
        Self {
            base_config: BenchmarkConfig::default(),
            frameworks_to_test: vec![
                "numpy".to_string(),
                "pytorch".to_string(),
                "tensorflow".to_string(),
            ],
            python_executable: "python3".to_string(),
            skip_missing_frameworks: true,
        }
    }
}

/// Check if a framework is available via Python
fn check_framework_availability(framework: &str, python_executable: &str) -> bool {
    let import_name = match framework {
        "numpy" => "numpy",
        "pytorch" => "torch",
        "tensorflow" => "tensorflow",
        _ => framework,
    };

    let output = Command::new(python_executable)
        .arg("-c")
        .arg(format!("import {import_name}"))
        .output();

    output.is_ok() && output.unwrap().status.success()
}

/// Generate Python benchmark script for a specific operation
fn generate_python_benchmark_script(
    framework: &str,
    operation: &str,
    size: usize,
    iterations: usize,
) -> String {
    let setup_code = match framework {
        "numpy" => format!(
            r#"
import numpy as np
import time
a = np.random.randn({size}).astype(np.float32)
b = np.random.randn({size}).astype(np.float32)
"#
        ),
        "pytorch" => format!(
            r#"
import torch
import time
a = torch.randn({size}, dtype=torch.float32)
b = torch.randn({size}, dtype=torch.float32)
"#
        ),
        "tensorflow" => format!(
            r#"
import tensorflow as tf
import time
a = tf.random.normal([{size}], dtype=tf.float32)
b = tf.random.normal([{size}], dtype=tf.float32)
"#
        ),
        _ => return String::new(),
    };

    let operation_code = match (framework, operation) {
        ("numpy", "add") => "result = np.add(a, b)",
        ("numpy", "mul") => "result = np.multiply(a, b)",
        ("numpy", "sub") => "result = np.subtract(a, b)",
        ("numpy", "div") => "result = np.divide(a, b)",
        ("pytorch", "add") => "result = torch.add(a, b)",
        ("pytorch", "mul") => "result = torch.mul(a, b)",
        ("pytorch", "sub") => "result = torch.sub(a, b)",
        ("pytorch", "div") => "result = torch.div(a, b)",
        ("tensorflow", "add") => "result = tf.add(a, b)",
        ("tensorflow", "mul") => "result = tf.multiply(a, b)",
        ("tensorflow", "sub") => "result = tf.subtract(a, b)",
        ("tensorflow", "div") => "result = tf.divide(a, b)",
        _ => return String::new(),
    };

    format!(
        r#"
{setup_code}

# Warmup
for _ in range(5):
    {operation_code}

# Benchmark
start_time = time.perf_counter()
for _ in range({iterations}):
    {operation_code}
end_time = time.perf_counter()

elapsed_ns = (end_time - start_time) * 1e9
print(f"{{elapsed_ns:.0f}}")
"#
    )
}

/// Benchmark a specific operation against external frameworks
fn benchmark_operation_against_frameworks(
    operation: &str,
    size: usize,
    config: &FrameworkBenchmarkConfig,
) -> Result<FrameworkComparisonResult> {
    // Benchmark TenfloweRS
    let tenflowers_time = benchmark_tenflowers_operation(operation, size, &config.base_config)?;

    // Benchmark other frameworks
    let mut framework_times = HashMap::new();

    for framework in &config.frameworks_to_test {
        if !check_framework_availability(framework, &config.python_executable) {
            if config.skip_missing_frameworks {
                println!("Warning: {framework} not available, skipping");
                continue;
            } else {
                return Err(TensorError::other(format!(
                    "Framework {framework} not available"
                )));
            }
        }

        if let Ok(time) = benchmark_external_framework(
            framework,
            operation,
            size,
            &config.base_config,
            &config.python_executable,
        ) {
            framework_times.insert(framework.clone(), time);
        } else {
            println!("Warning: Failed to benchmark {framework} for {operation}");
        }
    }

    Ok(FrameworkComparisonResult::new(
        operation.to_string(),
        size,
        tenflowers_time,
        framework_times,
    ))
}

/// Benchmark TenfloweRS operation
fn benchmark_tenflowers_operation(
    operation: &str,
    size: usize,
    config: &BenchmarkConfig,
) -> Result<Duration> {
    // Create test data
    let a_data: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..size).map(|i| (i as f32) + 1.0).collect();

    let a = Tensor::from_vec(a_data, &[size])?;
    let b = Tensor::from_vec(b_data, &[size])?;

    // Warmup
    for _ in 0..config.warmup_iterations {
        match operation {
            "add" => {
                let _ = super::binary::add(&a, &b)?;
            }
            "mul" => {
                let _ = super::binary::mul(&a, &b)?;
            }
            "sub" => {
                let _ = super::binary::sub(&a, &b)?;
            }
            "div" => {
                let _ = super::binary::div(&a, &b)?;
            }
            _ => {
                return Err(TensorError::other(format!(
                    "Unknown operation: {operation}"
                )))
            }
        }
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..config.measurement_iterations {
        match operation {
            "add" => {
                let _ = super::binary::add(&a, &b)?;
            }
            "mul" => {
                let _ = super::binary::mul(&a, &b)?;
            }
            "sub" => {
                let _ = super::binary::sub(&a, &b)?;
            }
            "div" => {
                let _ = super::binary::div(&a, &b)?;
            }
            _ => {
                return Err(TensorError::other(format!(
                    "Unknown operation: {operation}"
                )))
            }
        }
    }
    let elapsed = start.elapsed() / config.measurement_iterations as u32;

    Ok(elapsed)
}

/// Benchmark external framework via Python
fn benchmark_external_framework(
    framework: &str,
    operation: &str,
    size: usize,
    config: &BenchmarkConfig,
    python_executable: &str,
) -> Result<Duration> {
    let script =
        generate_python_benchmark_script(framework, operation, size, config.measurement_iterations);

    if script.is_empty() {
        return Err(TensorError::other(format!(
            "Unsupported framework/operation: {framework}/{operation}"
        )));
    }

    let output = Command::new(python_executable)
        .arg("-c")
        .arg(&script)
        .output()
        .map_err(|e| TensorError::other(format!("Failed to execute Python script: {e}")))?;

    if !output.status.success() {
        return Err(TensorError::other(format!(
            "Python script failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )));
    }

    let elapsed_ns_str = String::from_utf8_lossy(&output.stdout);
    let elapsed_ns: f64 = elapsed_ns_str
        .trim()
        .parse()
        .map_err(|e| TensorError::other(format!("Failed to parse timing result: {e}")))?;

    Ok(Duration::from_nanos(elapsed_ns as u64))
}

/// Run framework comparison benchmark suite
pub fn run_framework_comparison_benchmark(
    config: FrameworkBenchmarkConfig,
) -> Result<Vec<FrameworkComparisonResult>> {
    println!("Running TenfloweRS Framework Comparison Benchmark");
    println!("Testing against external frameworks...\n");

    let operations = vec!["add", "mul", "sub", "div"];
    let mut results = Vec::new();

    for &size in &config.base_config.sizes {
        println!("Benchmarking size: {size}");

        for operation in &operations {
            match benchmark_operation_against_frameworks(operation, size, &config) {
                Ok(result) => {
                    results.push(result);
                }
                Err(e) => {
                    println!("Warning: Failed to benchmark {operation} at size {size}: {e}");
                }
            }
        }
    }

    print_framework_comparison_results(&results);

    Ok(results)
}

/// Print framework comparison results in formatted table
pub fn print_framework_comparison_results(results: &[FrameworkComparisonResult]) {
    if results.is_empty() {
        println!("No benchmark results to display");
        return;
    }

    println!("\n{:-<120}", "");
    println!(
        "| {:^12} | {:^8} | {:^15} | {:^15} | {:^15} | {:^15} | {:^15} |",
        "Operation",
        "Size",
        "TenfloweRS (μs)",
        "NumPy (μs)",
        "PyTorch (μs)",
        "TensorFlow (μs)",
        "Relative Perf."
    );
    println!("{:-<120}", "");

    for result in results {
        let tf_us = result.tenflowers_time.as_micros();
        let numpy_us = result
            .framework_times
            .get("numpy")
            .map(|t| t.as_micros())
            .unwrap_or(0);
        let pytorch_us = result
            .framework_times
            .get("pytorch")
            .map(|t| t.as_micros())
            .unwrap_or(0);
        let tensorflow_us = result
            .framework_times
            .get("tensorflow")
            .map(|t| t.as_micros())
            .unwrap_or(0);

        // Calculate average relative performance (lower is better for TenfloweRS)
        let avg_relative = if !result.relative_performance.is_empty() {
            result.relative_performance.values().sum::<f64>()
                / result.relative_performance.len() as f64
        } else {
            0.0
        };

        println!(
            "| {:^12} | {:^8} | {:^15} | {:^15} | {:^15} | {:^15} | {:^15.2} |",
            result.operation,
            result.size,
            if tf_us > 0 {
                tf_us.to_string()
            } else {
                "-".to_string()
            },
            if numpy_us > 0 {
                numpy_us.to_string()
            } else {
                "-".to_string()
            },
            if pytorch_us > 0 {
                pytorch_us.to_string()
            } else {
                "-".to_string()
            },
            if tensorflow_us > 0 {
                tensorflow_us.to_string()
            } else {
                "-".to_string()
            },
            avg_relative
        );
    }
    println!("{:-<120}", "");

    // Summary statistics
    let all_relative_perfs: Vec<f64> = results
        .iter()
        .flat_map(|r| r.relative_performance.values())
        .cloned()
        .collect();

    if !all_relative_perfs.is_empty() {
        let avg_relative = all_relative_perfs.iter().sum::<f64>() / all_relative_perfs.len() as f64;
        let best_relative = all_relative_perfs
            .iter()
            .fold(f64::INFINITY, |a, &b| a.min(b));
        let worst_relative = all_relative_perfs.iter().fold(0.0f64, |a, &b| a.max(b));

        println!("Summary:");
        println!("  Average relative performance: {avg_relative:.2}x");
        println!("  Best relative performance: {best_relative:.2}x");
        println!("  Worst relative performance: {worst_relative:.2}x");

        if avg_relative < 1.0 {
            println!(
                "  🚀 TenfloweRS is on average {:.2}x faster than other frameworks",
                1.0 / avg_relative
            );
        } else {
            println!(
                "  ⚠️  TenfloweRS is on average {avg_relative:.2}x slower than other frameworks"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_framework_availability_check() {
        // Test with Python (should be available in most environments)
        let has_python = check_framework_availability("sys", "python3")
            || check_framework_availability("sys", "python");

        // This test is environment-dependent, so we just ensure it doesn't crash
        println!("Python available: {}", has_python);
    }

    #[test]
    fn test_benchmark_script_generation() {
        let script = generate_python_benchmark_script("numpy", "add", 1000, 10);
        assert!(script.contains("import numpy"));
        assert!(script.contains("np.add"));

        let script = generate_python_benchmark_script("pytorch", "mul", 1000, 10);
        assert!(script.contains("import torch"));
        assert!(script.contains("torch.mul"));
    }

    #[test]
    fn test_framework_comparison_result() {
        let mut framework_times = HashMap::new();
        framework_times.insert("numpy".to_string(), Duration::from_millis(2));
        framework_times.insert("pytorch".to_string(), Duration::from_millis(3));

        let result = FrameworkComparisonResult::new(
            "add".to_string(),
            1000,
            Duration::from_millis(1),
            framework_times,
        );

        assert_eq!(result.operation, "add");
        assert_eq!(result.size, 1000);
        assert!(result.relative_performance.contains_key("numpy"));
        assert!(result.relative_performance.contains_key("pytorch"));

        // TenfloweRS is faster (1ms vs 2ms, 3ms), so relative should be < 1.0
        assert!(result.relative_performance["numpy"] < 1.0);
        assert!(result.relative_performance["pytorch"] < 1.0);
    }
}
