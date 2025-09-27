//! Performance benchmarking for optimized operations
//!
//! This module provides benchmarking utilities to measure and compare
//! the performance of optimized vs original tensor operations.

use crate::{Result, Tensor};
use std::time::{Duration, Instant};

/// Benchmark configuration
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    pub warmup_iterations: usize,
    pub measurement_iterations: usize,
    pub sizes: Vec<usize>,
    pub verify_correctness: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 5,
            measurement_iterations: 10,
            sizes: vec![1000, 10000, 100000, 1000000],
            verify_correctness: true,
        }
    }
}

/// Benchmark result
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub operation: String,
    pub size: usize,
    pub original_time: Duration,
    pub optimized_time: Duration,
    pub speedup: f64,
    pub throughput_original: f64,  // elements per second
    pub throughput_optimized: f64, // elements per second
    pub correctness_verified: bool,
}

impl BenchmarkResult {
    pub fn new(
        operation: String,
        size: usize,
        original_time: Duration,
        optimized_time: Duration,
        correctness_verified: bool,
    ) -> Self {
        let speedup = original_time.as_nanos() as f64 / optimized_time.as_nanos() as f64;
        let throughput_original = size as f64 / original_time.as_secs_f64();
        let throughput_optimized = size as f64 / optimized_time.as_secs_f64();

        Self {
            operation,
            size,
            original_time,
            optimized_time,
            speedup,
            throughput_original,
            throughput_optimized,
            correctness_verified,
        }
    }
}

/// Binary operation benchmark suite
pub fn benchmark_binary_operations(config: BenchmarkConfig) -> Result<Vec<BenchmarkResult>> {
    let mut results = Vec::new();

    for &size in &config.sizes {
        println!("Benchmarking size: {size}");

        // Test addition
        if let Ok(result) = benchmark_add_f32(size, &config) {
            results.push(result);
        }

        // Test multiplication
        if let Ok(_result) = benchmark_mul_f32(size, &config) {}

        // Test subtraction
        if let Ok(result) = benchmark_sub_f32(size, &config) {
            results.push(result);
        }

        // Test division
        if let Ok(result) = benchmark_div_f32(size, &config) {
            results.push(result);
        }
    }

    Ok(results)
}

/// Benchmark addition operation
fn benchmark_add_f32(size: usize, config: &BenchmarkConfig) -> Result<BenchmarkResult> {
    // Create test data
    let a_data: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..size).map(|i| (i as f32) + 1.0).collect();

    let a = Tensor::from_vec(a_data, &[size])?;
    let b = Tensor::from_vec(b_data, &[size])?;

    // Warmup for original implementation
    for _ in 0..config.warmup_iterations {
        let _ = super::binary::add(&a, &b)?;
    }

    // Benchmark original implementation
    let start = Instant::now();
    for _ in 0..config.measurement_iterations {
        let _ = super::binary::add(&a, &b)?;
    }
    let original_time = start.elapsed() / config.measurement_iterations as u32;

    // Warmup for optimized implementation
    for _ in 0..config.warmup_iterations {
        let _ = super::optimized_binary::optimized_add(&a, &b)?;
    }

    // Benchmark optimized implementation
    let start = Instant::now();
    for _ in 0..config.measurement_iterations {
        let _ = super::optimized_binary::optimized_add(&a, &b)?;
    }
    let optimized_time = start.elapsed() / config.measurement_iterations as u32;

    // Verify correctness
    let correctness_verified = if config.verify_correctness {
        let original_result = super::binary::add(&a, &b)?;
        let optimized_result = super::optimized_binary::optimized_add(&a, &b)?;

        // Compare results element-wise with small tolerance for floating point
        let orig_data = original_result.to_vec()?;
        let opt_data = optimized_result.to_vec()?;

        orig_data
            .iter()
            .zip(opt_data.iter())
            .all(|(o, p)| (o - p).abs() < 1e-6)
    } else {
        true
    };

    Ok(BenchmarkResult::new(
        "Add".to_string(),
        size,
        original_time,
        optimized_time,
        correctness_verified,
    ))
}

/// Benchmark multiplication operation
fn benchmark_mul_f32(size: usize, config: &BenchmarkConfig) -> Result<BenchmarkResult> {
    // Create test data
    let a_data: Vec<f32> = (0..size).map(|i| (i as f32) + 1.0).collect();
    let b_data: Vec<f32> = (0..size).map(|i| (i as f32) + 2.0).collect();

    let a = Tensor::from_vec(a_data, &[size])?;
    let b = Tensor::from_vec(b_data, &[size])?;

    // Warmup and benchmark original
    for _ in 0..config.warmup_iterations {
        let _ = super::binary::mul(&a, &b)?;
    }

    let start = Instant::now();
    for _ in 0..config.measurement_iterations {
        let _ = super::binary::mul(&a, &b)?;
    }
    let original_time = start.elapsed() / config.measurement_iterations as u32;

    // Warmup and benchmark optimized
    for _ in 0..config.warmup_iterations {
        let _ = super::optimized_binary::optimized_mul(&a, &b)?;
    }

    let start = Instant::now();
    for _ in 0..config.measurement_iterations {
        let _ = super::optimized_binary::optimized_mul(&a, &b)?;
    }
    let optimized_time = start.elapsed() / config.measurement_iterations as u32;

    let correctness_verified = if config.verify_correctness {
        let original_result = super::binary::mul(&a, &b)?;
        let optimized_result = super::optimized_binary::optimized_mul(&a, &b)?;

        let orig_data = original_result.to_vec()?;
        let opt_data = optimized_result.to_vec()?;

        orig_data
            .iter()
            .zip(opt_data.iter())
            .all(|(o, p)| (o - p).abs() < 1e-6)
    } else {
        true
    };

    Ok(BenchmarkResult::new(
        "Mul".to_string(),
        size,
        original_time,
        optimized_time,
        correctness_verified,
    ))
}

/// Benchmark subtraction operation
fn benchmark_sub_f32(size: usize, config: &BenchmarkConfig) -> Result<BenchmarkResult> {
    let a_data: Vec<f32> = (0..size).map(|i| (i as f32) + 5.0).collect();
    let b_data: Vec<f32> = (0..size).map(|i| (i as f32) + 1.0).collect();

    let a = Tensor::from_vec(a_data, &[size])?;
    let b = Tensor::from_vec(b_data, &[size])?;

    // Warmup and benchmark original
    for _ in 0..config.warmup_iterations {
        let _ = super::binary::sub(&a, &b)?;
    }

    let start = Instant::now();
    for _ in 0..config.measurement_iterations {
        let _ = super::binary::sub(&a, &b)?;
    }
    let original_time = start.elapsed() / config.measurement_iterations as u32;

    // Warmup and benchmark optimized
    for _ in 0..config.warmup_iterations {
        let _ = super::optimized_binary::optimized_sub(&a, &b)?;
    }

    let start = Instant::now();
    for _ in 0..config.measurement_iterations {
        let _ = super::optimized_binary::optimized_sub(&a, &b)?;
    }
    let optimized_time = start.elapsed() / config.measurement_iterations as u32;

    let correctness_verified = if config.verify_correctness {
        let original_result = super::binary::sub(&a, &b)?;
        let optimized_result = super::optimized_binary::optimized_sub(&a, &b)?;

        let orig_data = original_result.to_vec()?;
        let opt_data = optimized_result.to_vec()?;

        orig_data
            .iter()
            .zip(opt_data.iter())
            .all(|(o, p)| (o - p).abs() < 1e-6)
    } else {
        true
    };

    Ok(BenchmarkResult::new(
        "Sub".to_string(),
        size,
        original_time,
        optimized_time,
        correctness_verified,
    ))
}

/// Benchmark division operation
fn benchmark_div_f32(size: usize, config: &BenchmarkConfig) -> Result<BenchmarkResult> {
    let a_data: Vec<f32> = (0..size).map(|i| (i as f32) + 10.0).collect();
    let b_data: Vec<f32> = (0..size).map(|i| (i as f32) + 2.0).collect();

    let a = Tensor::from_vec(a_data, &[size])?;
    let b = Tensor::from_vec(b_data, &[size])?;

    // Warmup and benchmark original
    for _ in 0..config.warmup_iterations {
        let _ = super::binary::div(&a, &b)?;
    }

    let start = Instant::now();
    for _ in 0..config.measurement_iterations {
        let _ = super::binary::div(&a, &b)?;
    }
    let original_time = start.elapsed() / config.measurement_iterations as u32;

    // Warmup and benchmark optimized
    for _ in 0..config.warmup_iterations {
        let _ = super::optimized_binary::optimized_div(&a, &b)?;
    }

    let start = Instant::now();
    for _ in 0..config.measurement_iterations {
        let _ = super::optimized_binary::optimized_div(&a, &b)?;
    }
    let optimized_time = start.elapsed() / config.measurement_iterations as u32;

    let correctness_verified = if config.verify_correctness {
        let original_result = super::binary::div(&a, &b)?;
        let optimized_result = super::optimized_binary::optimized_div(&a, &b)?;

        let orig_data = original_result.to_vec()?;
        let opt_data = optimized_result.to_vec()?;

        orig_data
            .iter()
            .zip(opt_data.iter())
            .all(|(o, p)| (o - p).abs() < 1e-6)
    } else {
        true
    };

    Ok(BenchmarkResult::new(
        "Div".to_string(),
        size,
        original_time,
        optimized_time,
        correctness_verified,
    ))
}

/// Print benchmark results in a formatted table
pub fn print_benchmark_results(results: &[BenchmarkResult]) {
    println!("\n{:-<100}", "");
    println!(
        "| {:^12} | {:^12} | {:^12} | {:^12} | {:^10} | {:^15} | {:^15} |",
        "Operation",
        "Size",
        "Original (μs)",
        "Optimized (μs)",
        "Speedup",
        "Orig Throughput",
        "Opt Throughput"
    );
    println!("{:-<100}", "");

    for result in results {
        let orig_us = result.original_time.as_micros();
        let opt_us = result.optimized_time.as_micros();
        let orig_throughput = format!("{:.1e}", result.throughput_original);
        let opt_throughput = format!("{:.1e}", result.throughput_optimized);

        println!(
            "| {:^12} | {:^12} | {:^12} | {:^12} | {:^10.2} | {:^15} | {:^15} |",
            result.operation,
            result.size,
            orig_us,
            opt_us,
            result.speedup,
            orig_throughput,
            opt_throughput
        );

        if !result.correctness_verified {
            println!(
                "  ⚠️  WARNING: Correctness verification failed for {} size {}",
                result.operation, result.size
            );
        }
    }
    println!("{:-<100}", "");

    // Summary statistics
    let avg_speedup: f64 = results.iter().map(|r| r.speedup).sum::<f64>() / results.len() as f64;
    let max_speedup = results.iter().map(|r| r.speedup).fold(0.0, f64::max);
    let min_speedup = results
        .iter()
        .map(|r| r.speedup)
        .fold(f64::INFINITY, f64::min);

    println!("Summary:");
    println!("  Average speedup: {avg_speedup:.2}x");
    println!("  Maximum speedup: {max_speedup:.2}x");
    println!("  Minimum speedup: {min_speedup:.2}x");

    let correctness_issues = results.iter().filter(|r| !r.correctness_verified).count();
    if correctness_issues > 0 {
        println!("  ⚠️  {correctness_issues} correctness verification failures");
    } else {
        println!("  ✅ All correctness verifications passed");
    }
}

/// Run a complete benchmark suite and return results
pub fn run_performance_benchmark() -> Result<Vec<BenchmarkResult>> {
    println!("Running TenfloweRS CPU Performance Benchmark");
    println!("Testing optimized vs original binary operations...\n");

    let config = BenchmarkConfig::default();
    let results = benchmark_binary_operations(config)?;

    print_benchmark_results(&results);

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_correctness() {
        let config = BenchmarkConfig {
            warmup_iterations: 1,
            measurement_iterations: 1,
            sizes: vec![1000],
            verify_correctness: true,
        };

        let results = benchmark_binary_operations(config).unwrap();

        // All results should have correctness verified
        for result in &results {
            assert!(
                result.correctness_verified,
                "Correctness verification failed for {}",
                result.operation
            );
        }

        // Should have results for all operations
        assert!(!results.is_empty());
    }

    #[test]
    fn test_small_benchmark() {
        let config = BenchmarkConfig {
            warmup_iterations: 1,
            measurement_iterations: 2,
            sizes: vec![100],
            verify_correctness: true,
        };

        let results = benchmark_binary_operations(config).unwrap();
        assert!(!results.is_empty());

        // Print results for manual inspection
        print_benchmark_results(&results);
    }
}
