//! TenfloweRS Performance Benchmark
//!
//! A comprehensive benchmarking utility to test TenfloweRS performance
//! across different operations, data types, and devices.
//!
//! Usage:
//!   cargo run --example tenflowers_benchmark
//!   cargo run --features gpu --example tenflowers_benchmark
//!   cargo run --features gpu,cuda --example tenflowers_benchmark

use std::collections::HashMap;
use std::time::{Duration, Instant};
use tenflowers_core::{
    ops::{add, conv2d, matmul, mul, random_normal_f32, relu, sigmoid, softmax, tanh},
    run_quick_health_check, run_system_health_check, DType, Device, Tensor,
};

#[derive(Debug)]
struct BenchmarkResult {
    operation: String,
    device: String,
    data_type: String,
    tensor_size: String,
    throughput_gflops: f64,
    latency_ms: f64,
    memory_usage_mb: Option<f64>,
}

#[derive(Debug)]
struct BenchmarkSuite {
    results: Vec<BenchmarkResult>,
    total_duration: Duration,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 TenfloweRS Performance Benchmark Suite");
    println!("==========================================\n");

    // Run system health check first
    println!("🔍 System Health Check:");
    println!("-----------------------");
    let health_result = run_quick_health_check();
    match health_result {
        Ok(info) => {
            println!("✅ System health: {:?}", info.health_status);
            println!("📦 Available devices: {}", info.available_devices.len());
            for device in &info.available_devices {
                println!("   • {}", device);
            }
        }
        Err(e) => {
            println!("⚠️  Health check failed: {}", e);
            println!("   Continuing with benchmark anyway...");
        }
    }

    println!("\n🏃 Starting Performance Benchmarks:");
    println!("====================================");

    let start_time = Instant::now();
    let mut benchmark_suite = BenchmarkSuite {
        results: Vec::new(),
        total_duration: Duration::new(0, 0),
    };

    // Test CPU operations
    println!("\n🖥️  CPU Benchmarks:");
    run_cpu_benchmarks(&mut benchmark_suite)?;

    // Test GPU operations if available
    #[cfg(feature = "gpu")]
    {
        println!("\n🔥 GPU Benchmarks:");
        if let Err(e) = run_gpu_benchmarks(&mut benchmark_suite) {
            println!("⚠️  GPU benchmarks failed: {}", e);
            println!("   This is normal if no GPU is available");
        }
    }

    #[cfg(not(feature = "gpu"))]
    {
        println!("\n🔥 GPU Benchmarks: Skipped (GPU feature not enabled)");
        println!("   Enable with: cargo run --features gpu --example tenflowers_benchmark");
    }

    // Neural network operations
    println!("\n🧠 Neural Network Operations:");
    run_neural_benchmarks(&mut benchmark_suite)?;

    benchmark_suite.total_duration = start_time.elapsed();

    // Print comprehensive results
    print_benchmark_results(&benchmark_suite);

    // Performance analysis
    analyze_performance(&benchmark_suite);

    println!("\n✨ Benchmark Complete! ✨");
    println!("Total benchmark time: {:?}", benchmark_suite.total_duration);

    Ok(())
}

fn run_cpu_benchmarks(suite: &mut BenchmarkSuite) -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;

    // Test different tensor sizes for arithmetic operations
    let sizes = vec![
        (vec![100], "Small (100)"),
        (vec![1000], "Medium (1K)"),
        (vec![10000], "Large (10K)"),
        (vec![100, 100], "Matrix (100x100)"),
        (vec![1000, 1000], "Large Matrix (1Kx1K)"),
    ];

    for (shape, size_desc) in sizes {
        // Addition benchmark
        benchmark_binary_op(suite, "CPU Add", &device, &shape, size_desc, |a, b| {
            add(a, b)
        })?;

        // Multiplication benchmark
        benchmark_binary_op(suite, "CPU Mul", &device, &shape, size_desc, |a, b| {
            mul(a, b)
        })?;

        // Matrix multiplication (for 2D tensors)
        if shape.len() == 2 {
            benchmark_binary_op(suite, "CPU MatMul", &device, &shape, size_desc, |a, b| {
                matmul(a, b)
            })?;
        }
    }

    Ok(())
}

#[cfg(feature = "gpu")]
fn run_gpu_benchmarks(suite: &mut BenchmarkSuite) -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::best_gpu()?;

    let sizes = vec![
        (vec![1000], "Medium (1K)"),
        (vec![10000], "Large (10K)"),
        (vec![100000], "Very Large (100K)"),
        (vec![1000, 1000], "Matrix (1Kx1K)"),
        (vec![2000, 2000], "Large Matrix (2Kx2K)"),
    ];

    for (shape, size_desc) in sizes {
        // GPU addition benchmark
        benchmark_binary_op(suite, "GPU Add", &device, &shape, size_desc, |a, b| {
            add(a, b)
        })?;

        // GPU multiplication benchmark
        benchmark_binary_op(suite, "GPU Mul", &device, &shape, size_desc, |a, b| {
            mul(a, b)
        })?;

        // GPU matrix multiplication (for 2D tensors)
        if shape.len() == 2 {
            benchmark_binary_op(suite, "GPU MatMul", &device, &shape, size_desc, |a, b| {
                matmul(a, b)
            })?;
        }
    }

    Ok(())
}

fn run_neural_benchmarks(suite: &mut BenchmarkSuite) -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;
    let batch_shapes = vec![
        (vec![32, 784], "Batch32x784 (MNIST-like)"),
        (vec![64, 512], "Batch64x512"),
        (vec![128, 256], "Batch128x256"),
    ];

    for (shape, size_desc) in batch_shapes {
        // ReLU activation
        benchmark_unary_op(suite, "ReLU", &device, &shape, size_desc, |a| relu(a))?;

        // Sigmoid activation
        benchmark_unary_op(suite, "Sigmoid", &device, &shape, size_desc, |a| sigmoid(a))?;

        // Tanh activation
        benchmark_unary_op(suite, "Tanh", &device, &shape, size_desc, |a| tanh(a))?;

        // Softmax (only for 2D tensors)
        if shape.len() == 2 {
            benchmark_unary_op(suite, "Softmax", &device, &shape, size_desc, |a| {
                softmax(a, Some(1))
            })?;
        }
    }

    // Convolution benchmark
    let conv_input_shape = vec![1, 3, 32, 32]; // Batch=1, Channels=3, Height=32, Width=32
    let conv_weight_shape = vec![16, 3, 3, 3]; // Out_channels=16, In_channels=3, Kernel=3x3

    println!("  🔄 Conv2D benchmark...");
    let start = Instant::now();

    let input = random_normal_f32(&conv_input_shape, 0.0, 1.0, None)?;
    let weight = random_normal_f32(&conv_weight_shape, 0.0, 1.0, None)?;

    let warmup_iterations = 3;
    for _ in 0..warmup_iterations {
        let _ = conv2d(&input, &weight, None, (1, 1), "valid")?;
    }

    let iterations = 10;
    let bench_start = Instant::now();
    for _ in 0..iterations {
        let _ = conv2d(&input, &weight, None, (1, 1), "valid")?;
    }
    let conv_time = bench_start.elapsed();
    let total_time = start.elapsed();

    let avg_latency = conv_time.as_secs_f64() * 1000.0 / iterations as f64;

    // Estimate FLOPS for convolution
    let batch_size = conv_input_shape[0];
    let out_channels = conv_weight_shape[0];
    let in_channels = conv_weight_shape[1];
    let kernel_h = conv_weight_shape[2];
    let kernel_w = conv_weight_shape[3];
    let out_h = conv_input_shape[2]; // Assuming same padding
    let out_w = conv_input_shape[3];

    let flops =
        (2 * batch_size * out_channels * in_channels * kernel_h * kernel_w * out_h * out_w) as f64;
    let gflops = (flops / avg_latency / 1000.0) / 1e9;

    suite.results.push(BenchmarkResult {
        operation: "Conv2D".to_string(),
        device: "CPU".to_string(),
        data_type: "f32".to_string(),
        tensor_size: "1x3x32x32 * 16x3x3x3".to_string(),
        throughput_gflops: gflops,
        latency_ms: avg_latency,
        memory_usage_mb: None,
    });

    println!(
        "    Conv2D: {:.2} GFLOPS, {:.3}ms latency",
        gflops, avg_latency
    );

    Ok(())
}

fn benchmark_binary_op<F>(
    suite: &mut BenchmarkSuite,
    op_name: &str,
    device: &Device,
    shape: &[usize],
    size_desc: &str,
    op: F,
) -> Result<(), Box<dyn std::error::Error>>
where
    F: Fn(&Tensor<f32>, &Tensor<f32>) -> Result<Tensor<f32>, tenflowers_core::TensorError>,
{
    println!("  🔄 {} ({})...", op_name, size_desc);

    let start = Instant::now();
    let a = Tensor::<f32>::ones(shape);
    let b = Tensor::<f32>::ones(shape);

    // Warmup
    let warmup_iterations = 3;
    for _ in 0..warmup_iterations {
        let _ = op(&a, &b)?;
    }

    // Benchmark
    let iterations = 50;
    let bench_start = Instant::now();
    for _ in 0..iterations {
        let _ = op(&a, &b)?;
    }
    let op_time = bench_start.elapsed();
    let total_time = start.elapsed();

    let avg_latency = op_time.as_secs_f64() * 1000.0 / iterations as f64;
    let elements = shape.iter().product::<usize>() as f64;

    // Calculate GFLOPS (operations per second in billions)
    let ops_per_element = if op_name.contains("MatMul") && shape.len() == 2 {
        2.0 * shape[1] as f64 // Matrix multiplication: 2N operations per output element
    } else {
        1.0 // Element-wise operations: 1 operation per element
    };

    let total_ops = elements * ops_per_element;
    let gflops = (total_ops / avg_latency / 1000.0) / 1e9;

    suite.results.push(BenchmarkResult {
        operation: op_name.to_string(),
        device: device.to_string(),
        data_type: "f32".to_string(),
        tensor_size: size_desc.to_string(),
        throughput_gflops: gflops,
        latency_ms: avg_latency,
        memory_usage_mb: None,
    });

    println!(
        "    {}: {:.2} GFLOPS, {:.3}ms latency",
        op_name, gflops, avg_latency
    );

    Ok(())
}

fn benchmark_unary_op<F>(
    suite: &mut BenchmarkSuite,
    op_name: &str,
    device: &Device,
    shape: &[usize],
    size_desc: &str,
    op: F,
) -> Result<(), Box<dyn std::error::Error>>
where
    F: Fn(&Tensor<f32>) -> Result<Tensor<f32>, tenflowers_core::TensorError>,
{
    println!("  🔄 {} ({})...", op_name, size_desc);

    let start = Instant::now();
    let a = random_normal_f32(shape, 0.0, 1.0, None)?;

    // Warmup
    let warmup_iterations = 3;
    for _ in 0..warmup_iterations {
        let _ = op(&a)?;
    }

    // Benchmark
    let iterations = 20;
    let bench_start = Instant::now();
    for _ in 0..iterations {
        let _ = op(&a)?;
    }
    let op_time = bench_start.elapsed();
    let total_time = start.elapsed();

    let avg_latency = op_time.as_secs_f64() * 1000.0 / iterations as f64;
    let elements = shape.iter().product::<usize>() as f64;
    let gflops = (elements / avg_latency / 1000.0) / 1e9;

    suite.results.push(BenchmarkResult {
        operation: op_name.to_string(),
        device: device.to_string(),
        data_type: "f32".to_string(),
        tensor_size: size_desc.to_string(),
        throughput_gflops: gflops,
        latency_ms: avg_latency,
        memory_usage_mb: None,
    });

    println!(
        "    {}: {:.2} GFLOPS, {:.3}ms latency",
        op_name, gflops, avg_latency
    );

    Ok(())
}

fn print_benchmark_results(suite: &BenchmarkSuite) {
    println!("\n📊 Comprehensive Benchmark Results:");
    println!("===================================");

    // Group results by operation type
    let mut grouped_results: HashMap<String, Vec<&BenchmarkResult>> = HashMap::new();
    for result in &suite.results {
        let key = result
            .operation
            .split_whitespace()
            .last()
            .unwrap_or(&result.operation)
            .to_string();
        grouped_results
            .entry(key)
            .or_insert_with(Vec::new)
            .push(result);
    }

    for (op_type, results) in grouped_results {
        println!("\n🔹 {} Operations:", op_type);
        println!(
            "  {:<15} {:<10} {:<20} {:<12} {:<10}",
            "Device", "Data Type", "Tensor Size", "GFLOPS", "Latency(ms)"
        );
        println!("  {}", "-".repeat(70));

        for result in results {
            println!(
                "  {:<15} {:<10} {:<20} {:<12.2} {:<10.3}",
                result.device,
                result.data_type,
                result.tensor_size,
                result.throughput_gflops,
                result.latency_ms
            );
        }
    }

    // Summary statistics
    println!("\n📈 Summary Statistics:");
    println!("======================");

    let total_operations = suite.results.len();
    let avg_gflops = suite
        .results
        .iter()
        .map(|r| r.throughput_gflops)
        .sum::<f64>()
        / total_operations as f64;
    let max_gflops = suite
        .results
        .iter()
        .map(|r| r.throughput_gflops)
        .fold(0.0, f64::max);
    let min_latency = suite
        .results
        .iter()
        .map(|r| r.latency_ms)
        .fold(f64::INFINITY, f64::min);

    println!("  Total operations benchmarked: {}", total_operations);
    println!("  Average throughput: {:.2} GFLOPS", avg_gflops);
    println!("  Peak throughput: {:.2} GFLOPS", max_gflops);
    println!("  Minimum latency: {:.3} ms", min_latency);
    println!("  Total benchmark duration: {:?}", suite.total_duration);
}

fn analyze_performance(suite: &BenchmarkSuite) {
    println!("\n🔍 Performance Analysis:");
    println!("========================");

    // Find best performing operations
    let best_throughput = suite.results.iter().max_by(|a, b| {
        a.throughput_gflops
            .partial_cmp(&b.throughput_gflops)
            .unwrap()
    });

    let fastest_operation = suite
        .results
        .iter()
        .min_by(|a, b| a.latency_ms.partial_cmp(&b.latency_ms).unwrap());

    if let Some(best) = best_throughput {
        println!(
            "🏆 Highest throughput: {} on {} ({:.2} GFLOPS)",
            best.operation, best.device, best.throughput_gflops
        );
    }

    if let Some(fastest) = fastest_operation {
        println!(
            "⚡ Fastest operation: {} on {} ({:.3} ms)",
            fastest.operation, fastest.device, fastest.latency_ms
        );
    }

    // Performance recommendations
    println!("\n💡 Recommendations:");

    let cpu_results: Vec<_> = suite
        .results
        .iter()
        .filter(|r| r.device.contains("CPU"))
        .collect();
    let gpu_results: Vec<_> = suite
        .results
        .iter()
        .filter(|r| r.device.contains("GPU"))
        .collect();

    if !gpu_results.is_empty() && !cpu_results.is_empty() {
        let avg_cpu_gflops =
            cpu_results.iter().map(|r| r.throughput_gflops).sum::<f64>() / cpu_results.len() as f64;
        let avg_gpu_gflops =
            gpu_results.iter().map(|r| r.throughput_gflops).sum::<f64>() / gpu_results.len() as f64;

        if avg_gpu_gflops > avg_cpu_gflops {
            let speedup = avg_gpu_gflops / avg_cpu_gflops;
            println!("  🚀 GPU provides {:.1}x average speedup over CPU", speedup);
            println!("     Consider using GPU for large tensor operations");
        } else {
            println!("  💻 CPU performance is competitive with GPU for these workloads");
        }
    } else if gpu_results.is_empty() {
        println!("  🔥 Enable GPU support for potentially better performance:");
        println!("     cargo run --features gpu --example tenflowers_benchmark");
    }

    if suite.total_duration.as_secs() > 30 {
        println!(
            "  ⏱️  Benchmark took {}s - performance looks good!",
            suite.total_duration.as_secs()
        );
    } else {
        println!("  ⚡ Quick benchmark completion - excellent system performance!");
    }
}
