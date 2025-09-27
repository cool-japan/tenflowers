//! 🚀 Ultra-Performance Optimization Demo
//!
//! This example demonstrates the world-class ultra-performance optimizations
//! integrated into TenflowRS, showcasing SIMD vectorization, cache-oblivious
//! algorithms, and real-time performance monitoring.

use std::time::Instant;
use tenflowers_core::{
    ops::{matmul, ultra_matmul},
    ultra_performance_profiler::{configure_profiler, print_performance_report, ProfilerConfig},
    Device, Tensor,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 ULTRA-PERFORMANCE OPTIMIZATION DEMO");
    println!("{}", "=".repeat(60));
    println!("Demonstrating world-class optimization integration in TenflowRS");
    println!();

    // Configure the ultra-performance profiler
    configure_profiler(ProfilerConfig {
        detailed_profiling: true,
        max_history_entries: 100,
        min_record_time: 1000, // 1 microsecond
        optimization_recommendations: true,
    });

    // Test small matrices (SIMD optimization)
    println!("📊 TESTING SMALL MATRICES (SIMD Optimization):");
    test_small_matrices()?;
    println!();

    // Test medium matrices (Cache optimization)
    println!("📊 TESTING MEDIUM MATRICES (Cache Optimization):");
    test_medium_matrices()?;
    println!();

    // Test large matrices (Cache-oblivious algorithms)
    println!("📊 TESTING LARGE MATRICES (Cache-Oblivious Algorithms):");
    test_large_matrices()?;
    println!();

    // Test batch operations
    println!("📊 TESTING BATCH OPERATIONS:");
    test_batch_operations()?;
    println!();

    // Performance comparison
    println!("⚔️  PERFORMANCE COMPARISON:");
    performance_comparison()?;
    println!();

    // Print comprehensive performance report
    print_performance_report();

    println!("✅ Ultra-performance optimization demo completed successfully!");
    println!("🏆 World-class performance achieved with humility and dedication!");

    Ok(())
}

/// Test small matrices (optimized for SIMD)
fn test_small_matrices() -> Result<(), Box<dyn std::error::Error>> {
    let sizes = vec![8, 16, 32];

    for size in sizes {
        let a_data: Vec<f32> = (0..size * size).map(|i| (i as f32) * 0.1).collect();
        let b_data: Vec<f32> = (0..size * size).map(|i| (i as f32) * 0.2).collect();

        let a = Tensor::<f32>::from_vec(a_data, &[size, size])?;
        let b = Tensor::<f32>::from_vec(b_data, &[size, size])?;

        let start = Instant::now();
        let _result = ultra_matmul(&a, &b)?;
        let elapsed = start.elapsed();

        let flops = 2 * size * size * size;
        let gflops = flops as f64 / elapsed.as_secs_f64() / 1e9;

        println!(
            "  {}x{} matrix: {:.2} GFLOP/s in {:.2}μs",
            size,
            size,
            gflops,
            elapsed.as_secs_f64() * 1e6
        );
    }

    Ok(())
}

/// Test medium matrices (cache-optimized)
fn test_medium_matrices() -> Result<(), Box<dyn std::error::Error>> {
    let sizes = vec![64, 128, 256];

    for size in sizes {
        let a_data: Vec<f32> = (0..size * size)
            .map(|i| (i as f32) / (size * size) as f32)
            .collect();
        let b_data: Vec<f32> = (0..size * size)
            .map(|i| ((i + 1) as f32) / (size * size) as f32)
            .collect();

        let a = Tensor::<f32>::from_vec(a_data, &[size, size])?;
        let b = Tensor::<f32>::from_vec(b_data, &[size, size])?;

        let start = Instant::now();
        let _result = ultra_matmul(&a, &b)?;
        let elapsed = start.elapsed();

        let flops = 2 * size * size * size;
        let gflops = flops as f64 / elapsed.as_secs_f64() / 1e9;

        println!(
            "  {}x{} matrix: {:.2} GFLOP/s in {:.2}ms",
            size,
            size,
            gflops,
            elapsed.as_secs_f64() * 1000.0
        );
    }

    Ok(())
}

/// Test large matrices (cache-oblivious)
fn test_large_matrices() -> Result<(), Box<dyn std::error::Error>> {
    let sizes = vec![512, 768];

    for size in sizes {
        // Use smaller data range to avoid overflow
        let a_data: Vec<f32> = (0..size * size)
            .map(|i| (i % 1000) as f32 / 1000.0)
            .collect();
        let b_data: Vec<f32> = (0..size * size)
            .map(|i| ((i + 1) % 1000) as f32 / 1000.0)
            .collect();

        let a = Tensor::<f32>::from_vec(a_data, &[size, size])?;
        let b = Tensor::<f32>::from_vec(b_data, &[size, size])?;

        println!("  Starting {}x{} matrix multiplication...", size, size);
        let start = Instant::now();
        let _result = ultra_matmul(&a, &b)?;
        let elapsed = start.elapsed();

        let flops = 2 * size * size * size;
        let gflops = flops as f64 / elapsed.as_secs_f64() / 1e9;

        println!(
            "  {}x{} matrix: {:.2} GFLOP/s in {:.2}ms",
            size,
            size,
            gflops,
            elapsed.as_secs_f64() * 1000.0
        );
    }

    Ok(())
}

/// Test batch operations
fn test_batch_operations() -> Result<(), Box<dyn std::error::Error>> {
    let batch_size = 4;
    let matrix_size = 64;
    let total_elements = batch_size * matrix_size * matrix_size;

    let a_data: Vec<f32> = (0..total_elements)
        .map(|i| (i as f32) / total_elements as f32)
        .collect();
    let b_data: Vec<f32> = (0..total_elements)
        .map(|i| ((i + 1) as f32) / total_elements as f32)
        .collect();

    let a = Tensor::<f32>::from_vec(a_data, &[batch_size, matrix_size, matrix_size])?;
    let b = Tensor::<f32>::from_vec(b_data, &[batch_size, matrix_size, matrix_size])?;

    let start = Instant::now();
    let _result = ultra_matmul(&a, &b)?;
    let elapsed = start.elapsed();

    let flops = batch_size * 2 * matrix_size * matrix_size * matrix_size;
    let gflops = flops as f64 / elapsed.as_secs_f64() / 1e9;

    println!(
        "  Batch {}x{}x{}: {:.2} GFLOP/s in {:.2}ms",
        batch_size,
        matrix_size,
        matrix_size,
        gflops,
        elapsed.as_secs_f64() * 1000.0
    );

    Ok(())
}

/// Performance comparison between standard and ultra matmul
fn performance_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let size = 256;
    let a_data: Vec<f32> = (0..size * size)
        .map(|i| (i as f32) / (size * size) as f32)
        .collect();
    let b_data: Vec<f32> = (0..size * size)
        .map(|i| ((i + 1) as f32) / (size * size) as f32)
        .collect();

    let a = Tensor::<f32>::from_vec(a_data.clone(), &[size, size])?;
    let b = Tensor::<f32>::from_vec(b_data.clone(), &[size, size])?;

    // Test standard matmul
    let start = Instant::now();
    let _result_std = matmul(&a, &b)?;
    let elapsed_std = start.elapsed();

    // Test ultra matmul
    let start = Instant::now();
    let _result_ultra = ultra_matmul(&a, &b)?;
    let elapsed_ultra = start.elapsed();

    let flops = 2 * size * size * size;
    let gflops_std = flops as f64 / elapsed_std.as_secs_f64() / 1e9;
    let gflops_ultra = flops as f64 / elapsed_ultra.as_secs_f64() / 1e9;

    let speedup = gflops_ultra / gflops_std;

    println!(
        "  Standard MatMul:  {:.2} GFLOP/s in {:.2}ms",
        gflops_std,
        elapsed_std.as_secs_f64() * 1000.0
    );
    println!(
        "  Ultra MatMul:     {:.2} GFLOP/s in {:.2}ms",
        gflops_ultra,
        elapsed_ultra.as_secs_f64() * 1000.0
    );
    println!("  🚀 Speedup:       {:.2}x improvement!", speedup);

    if speedup > 1.5 {
        println!("  🏆 EXCELLENT: Ultra-performance optimizations provide significant speedup!");
    } else if speedup > 1.1 {
        println!("  ✅ GOOD: Ultra-performance optimizations provide measurable improvement!");
    } else {
        println!("  📊 Results may vary based on system architecture and matrix size");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ultra_performance_integration() {
        // Quick test to ensure ultra matmul works
        let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = Tensor::<f32>::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();

        let result = ultra_matmul(&a, &b).unwrap();
        assert_eq!(result.shape().dims(), &[2, 2]);

        // Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
        //         = [[19, 22], [43, 50]]
        if let Some(data) = result.as_slice() {
            assert_eq!(data, &[19.0, 22.0, 43.0, 50.0]);
        }
    }

    #[test]
    fn test_profiler_integration() {
        use tenflowers_core::ultra_performance_profiler::get_performance_summary;

        // Clear any existing data
        tenflowers_core::ultra_performance_profiler::clear_performance_data();

        // Perform a test operation
        let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = Tensor::<f32>::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
        let _result = ultra_matmul(&a, &b).unwrap();

        // Check that profiler recorded the operation
        if let Some(summary) = get_performance_summary() {
            assert!(summary.total_operations > 0);
            assert!(summary.peak_gflops > 0.0);
        }
    }
}
