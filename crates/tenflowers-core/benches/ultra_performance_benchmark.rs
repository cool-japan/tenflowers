//! 🚀 Ultra-Performance Comprehensive Benchmarking Suite
//!
//! This benchmark suite provides detailed performance analysis and validation
//! of the ultra-performance optimizations integrated into TenflowRS.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::time::Duration;
use tenflowers_core::{
    ops::{matmul, ultra_matmul},
    ultra_performance_profiler::{configure_profiler, print_performance_report, ProfilerConfig},
    Tensor,
};

/// Benchmark matrix multiplication performance across different sizes
pub fn benchmark_matmul_sizes(c: &mut Criterion) {
    // Configure profiler for detailed analysis
    configure_profiler(ProfilerConfig {
        detailed_profiling: true,
        max_history_entries: 1000,
        min_record_time: 100, // 100 nanoseconds
        optimization_recommendations: true,
    });

    let mut group = c.benchmark_group("matmul_sizes");

    // Test different matrix sizes to evaluate optimization effectiveness
    let sizes = vec![8, 16, 32, 64, 128, 256, 512];

    for size in sizes {
        let flops = 2 * size * size * size; // FMAOPs for matrix multiplication
        group.throughput(Throughput::Elements(flops as u64));

        // Generate test data
        let a_data: Vec<f32> = (0..size * size)
            .map(|i| (i as f32) / (size as f32))
            .collect();
        let b_data: Vec<f32> = (0..size * size)
            .map(|i| ((i + 1) as f32) / (size as f32))
            .collect();

        let a = Tensor::<f32>::from_vec(a_data.clone(), &[size, size]).unwrap();
        let b = Tensor::<f32>::from_vec(b_data.clone(), &[size, size]).unwrap();

        // Benchmark standard matmul
        group.bench_with_input(
            BenchmarkId::new("standard_matmul", size),
            &size,
            |bencher, _| {
                bencher.iter(|| {
                    let result = matmul(black_box(&a), black_box(&b)).unwrap();
                    black_box(result);
                });
            },
        );

        // Benchmark ultra-performance matmul
        group.bench_with_input(
            BenchmarkId::new("ultra_matmul", size),
            &size,
            |bencher, _| {
                bencher.iter(|| {
                    let result = ultra_matmul(black_box(&a), black_box(&b)).unwrap();
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark different matrix aspect ratios
pub fn benchmark_matmul_aspect_ratios(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_aspect_ratios");

    // Test different aspect ratios to evaluate optimization strategies
    let test_cases = vec![
        ("square_small", 64, 64, 64),
        ("wide_matrix", 32, 128, 64),
        ("tall_matrix", 128, 32, 64),
        ("outer_product", 128, 1, 128),
        ("inner_product", 1, 128, 1),
        ("extreme_wide", 16, 256, 32),
        ("extreme_tall", 256, 16, 32),
    ];

    for (name, m, k, n) in test_cases {
        let flops = 2 * m * k * n;
        group.throughput(Throughput::Elements(flops as u64));

        // Generate test data
        let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32) / (m * k) as f32).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32) / (k * n) as f32).collect();

        let a = Tensor::<f32>::from_vec(a_data, &[m, k]).unwrap();
        let b = Tensor::<f32>::from_vec(b_data, &[k, n]).unwrap();

        group.bench_function(name, |bencher| {
            bencher.iter(|| {
                let result = ultra_matmul(black_box(&a), black_box(&b)).unwrap();
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark batch matrix operations
pub fn benchmark_batch_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_operations");

    // Test different batch configurations
    let batch_configs = vec![
        ("small_batch", 2, 32, 32, 32),
        ("medium_batch", 4, 64, 64, 64),
        ("large_batch", 8, 32, 32, 32),
        ("many_small", 16, 16, 16, 16),
    ];

    for (name, batch_size, m, k, n) in batch_configs {
        let total_elements = batch_size * m * k * n;
        let flops = batch_size * 2 * m * k * n;
        group.throughput(Throughput::Elements(flops as u64));

        let a_data: Vec<f32> = (0..batch_size * m * k)
            .map(|i| (i as f32) / total_elements as f32)
            .collect();
        let b_data: Vec<f32> = (0..batch_size * k * n)
            .map(|i| (i as f32) / total_elements as f32)
            .collect();

        let a = Tensor::<f32>::from_vec(a_data, &[batch_size, m, k]).unwrap();
        let b = Tensor::<f32>::from_vec(b_data, &[batch_size, k, n]).unwrap();

        group.bench_function(name, |bencher| {
            bencher.iter(|| {
                let result = ultra_matmul(black_box(&a), black_box(&b)).unwrap();
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark memory access patterns
pub fn benchmark_memory_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_patterns");
    group.measurement_time(Duration::from_secs(10));

    let size = 128;
    let flops = 2 * size * size * size;
    group.throughput(Throughput::Elements(flops as u64));

    // Test contiguous vs non-contiguous memory access
    let a_data: Vec<f32> = (0..size * size).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..size * size).map(|i| (i + 1) as f32).collect();

    let a = Tensor::<f32>::from_vec(a_data, &[size, size]).unwrap();
    let b = Tensor::<f32>::from_vec(b_data, &[size, size]).unwrap();

    group.bench_function("contiguous_memory", |bencher| {
        bencher.iter(|| {
            let result = ultra_matmul(black_box(&a), black_box(&b)).unwrap();
            black_box(result);
        });
    });

    // Test with different data patterns that stress different cache levels
    group.bench_function("cache_stress_l1", |bencher| {
        // Small matrices that fit in L1 cache
        let small_size = 32;
        let small_a_data: Vec<f32> = (0..small_size * small_size).map(|i| i as f32).collect();
        let small_b_data: Vec<f32> = (0..small_size * small_size)
            .map(|i| (i + 1) as f32)
            .collect();

        let small_a = Tensor::<f32>::from_vec(small_a_data, &[small_size, small_size]).unwrap();
        let small_b = Tensor::<f32>::from_vec(small_b_data, &[small_size, small_size]).unwrap();

        bencher.iter(|| {
            let result = ultra_matmul(black_box(&small_a), black_box(&small_b)).unwrap();
            black_box(result);
        });
    });

    group.bench_function("cache_stress_l2", |bencher| {
        // Medium matrices that fit in L2 cache but not L1
        let medium_size = 64;
        let medium_a_data: Vec<f32> = (0..medium_size * medium_size).map(|i| i as f32).collect();
        let medium_b_data: Vec<f32> = (0..medium_size * medium_size)
            .map(|i| (i + 1) as f32)
            .collect();

        let medium_a = Tensor::<f32>::from_vec(medium_a_data, &[medium_size, medium_size]).unwrap();
        let medium_b = Tensor::<f32>::from_vec(medium_b_data, &[medium_size, medium_size]).unwrap();

        bencher.iter(|| {
            let result = ultra_matmul(black_box(&medium_a), black_box(&medium_b)).unwrap();
            black_box(result);
        });
    });

    group.finish();
}

/// Benchmark optimization strategy selection
pub fn benchmark_optimization_strategies(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimization_strategies");

    // Test matrices that trigger different optimization paths
    let strategy_tests = vec![
        ("micro_matrices", 4, 4, 4),       // Should trigger micro SIMD
        ("small_matrices", 16, 16, 16),    // Should trigger small cache-optimized
        ("medium_matrices", 64, 64, 64),   // Should trigger medium adaptive
        ("large_matrices", 256, 256, 256), // Should trigger cache-oblivious
        ("outer_product", 100, 1, 100),    // Should trigger outer product optimization
    ];

    for (name, m, k, n) in strategy_tests {
        let flops = 2 * m * k * n;
        group.throughput(Throughput::Elements(flops as u64));

        let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32) / (m * k) as f32).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32) / (k * n) as f32).collect();

        let a = Tensor::<f32>::from_vec(a_data, &[m, k]).unwrap();
        let b = Tensor::<f32>::from_vec(b_data, &[k, n]).unwrap();

        group.bench_function(name, |bencher| {
            bencher.iter(|| {
                let result = ultra_matmul(black_box(&a), black_box(&b)).unwrap();
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark data type performance
pub fn benchmark_data_types(c: &mut Criterion) {
    let mut group = c.benchmark_group("data_types");

    let size = 64;
    let flops = 2 * size * size * size;
    group.throughput(Throughput::Elements(flops as u64));

    // Test f32 performance (optimized path)
    group.bench_function("f32_optimized", |bencher| {
        let a_data: Vec<f32> = (0..size * size).map(|i| i as f32).collect();
        let b_data: Vec<f32> = (0..size * size).map(|i| (i + 1) as f32).collect();

        let a = Tensor::<f32>::from_vec(a_data, &[size, size]).unwrap();
        let b = Tensor::<f32>::from_vec(b_data, &[size, size]).unwrap();

        bencher.iter(|| {
            let result = ultra_matmul(black_box(&a), black_box(&b)).unwrap();
            black_box(result);
        });
    });

    // Test f64 performance (generic path)
    group.bench_function("f64_generic", |bencher| {
        let a_data: Vec<f64> = (0..size * size).map(|i| i as f64).collect();
        let b_data: Vec<f64> = (0..size * size).map(|i| (i + 1) as f64).collect();

        let a = Tensor::<f64>::from_vec(a_data, &[size, size]).unwrap();
        let b = Tensor::<f64>::from_vec(b_data, &[size, size]).unwrap();

        bencher.iter(|| {
            let result = ultra_matmul(black_box(&a), black_box(&b)).unwrap();
            black_box(result);
        });
    });

    group.finish();
}

/// Performance scaling analysis
pub fn benchmark_performance_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("performance_scaling");
    group.measurement_time(Duration::from_secs(15));

    // Test how performance scales with matrix size
    let scaling_sizes = vec![32, 64, 96, 128, 160, 192, 224, 256];

    for size in scaling_sizes {
        let flops = 2 * size * size * size;
        group.throughput(Throughput::Elements(flops as u64));

        let a_data: Vec<f32> = (0..size * size)
            .map(|i| (i as f32) / (size as f32))
            .collect();
        let b_data: Vec<f32> = (0..size * size)
            .map(|i| ((i + 1) as f32) / (size as f32))
            .collect();

        let a = Tensor::<f32>::from_vec(a_data, &[size, size]).unwrap();
        let b = Tensor::<f32>::from_vec(b_data, &[size, size]).unwrap();

        group.bench_with_input(BenchmarkId::new("scaling", size), &size, |bencher, _| {
            bencher.iter(|| {
                let result = ultra_matmul(black_box(&a), black_box(&b)).unwrap();
                black_box(result);
            });
        });
    }

    group.finish();

    // Print comprehensive performance report after benchmarks
    println!("\n🚀 ULTRA-PERFORMANCE BENCHMARK ANALYSIS COMPLETE");
    println!("{}", "=".repeat(70));
    print_performance_report();
}

criterion_group!(
    ultra_performance_benches,
    benchmark_matmul_sizes,
    benchmark_matmul_aspect_ratios,
    benchmark_batch_operations,
    benchmark_memory_patterns,
    benchmark_optimization_strategies,
    benchmark_data_types,
    benchmark_performance_scaling
);

criterion_main!(ultra_performance_benches);
