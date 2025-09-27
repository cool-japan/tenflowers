use std::time::Duration;
use tenflowers_core::{
    ops::benchmark::{
        benchmark_binary_op, benchmark_matmul_sizes, benchmark_unary_op, BenchmarkConfig,
        BenchmarkSuite,
    },
    DType, Device, Tensor,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("TenfloweRS Benchmark Suite");
    println!("=========================\n");

    // Example 1: Basic benchmark configuration
    println!("1. Creating benchmark suite...");
    let config = BenchmarkConfig {
        warmup_iterations: 5,
        measurement_iterations: 50,
        measure_memory: true,
        calculate_flops: true,
        min_execution_time: Duration::from_millis(1),
        max_execution_time: Duration::from_secs(10),
    };

    let suite = BenchmarkSuite::new(config);
    println!("Benchmark suite created!\n");

    // Example 2: Binary operation benchmarks
    println!("2. Running binary operation benchmarks...");
    let devices = vec![Device::Cpu];
    #[cfg(feature = "gpu")]
    let devices = vec![Device::Cpu, Device::Gpu(0)];

    let dtypes = vec![DType::Float32, DType::Float64];

    for shape in &[[100, 100], [500, 500], [1000, 1000]] {
        println!("  Benchmarking shape {:?}...", shape);

        // Benchmark Add operation
        let add_results = benchmark_binary_op::<f32>("Add", shape, shape, &devices, &dtypes)?;
        for result in add_results {
            println!(
                "    Add on {:?}: {:.2} μs",
                result.device,
                result.duration.as_micros()
            );
        }

        // Benchmark Mul operation
        let mul_results = benchmark_binary_op::<f32>("Mul", shape, shape, &devices, &dtypes)?;
        for result in mul_results {
            println!(
                "    Mul on {:?}: {:.2} μs",
                result.device,
                result.duration.as_micros()
            );
        }
    }
    println!("Binary operation benchmarks completed!\n");

    // Example 3: Matrix multiplication benchmarks
    println!("3. Running matrix multiplication benchmarks...");
    let matmul_sizes = vec![(64, 128, 256), (128, 256, 512), (256, 512, 1024)];
    let matmul_results = benchmark_matmul_sizes::<f32>(&matmul_sizes, &devices, &dtypes)?;

    for result in matmul_results {
        println!(
            "  MatMul {:?} on {:?}: {:.2} μs, {:.2e} FLOPS",
            result.input_shapes,
            result.device,
            result.duration.as_micros(),
            result.flops.unwrap_or(0.0)
        );
    }
    println!("Matrix multiplication benchmarks completed!\n");

    // Example 4: Unary operation benchmarks
    println!("4. Running unary operation benchmarks...");
    let unary_shapes = vec![[1000], [10000], [100000]];

    for shape in unary_shapes {
        println!("  Benchmarking unary ops with shape {:?}...", shape);
        let relu_results = benchmark_unary_op::<f32>("ReLU", &shape, &devices, &dtypes)?;
        for result in relu_results {
            println!(
                "    ReLU on {:?}: {:.2} μs",
                result.device,
                result.duration.as_micros()
            );
        }
    }
    println!("Unary operation benchmarks completed!\n");

    // Example 5: Generate comprehensive report
    println!("5. Generating performance report...");
    let report = suite.generate_report();
    println!("{}", report);

    // Example 6: Create a simple tensor benchmark manually
    println!("6. Running custom tensor benchmark...");
    let tensor_a = Tensor::<f32>::ones(&[256, 256]);
    let tensor_b = Tensor::<f32>::ones(&[256, 256]);

    let attrs = std::collections::HashMap::new();
    let inputs = vec![&tensor_a, &tensor_b];

    match suite.benchmark_operation("Add", &inputs, &attrs) {
        Ok(result) => {
            println!(
                "  Custom Add benchmark: {:.2} μs, throughput: {:.2e} elem/s",
                result.duration.as_micros(),
                result.throughput.unwrap_or(0.0)
            );
        }
        Err(e) => {
            println!("  Custom benchmark failed: {}", e);
        }
    }

    println!("\nAll benchmark examples completed successfully!");

    Ok(())
}

/// Example of benchmark result analysis
#[allow(dead_code)]
fn analyze_benchmark_results(report: String) {
    println!("Analyzing benchmark results...");

    // Extract performance metrics
    let lines: Vec<&str> = report.lines().collect();

    for line in lines {
        if line.contains("GPU") && line.contains("CPU") {
            println!("Performance comparison found: {}", line);
        }

        if line.contains("Duration") {
            println!("Performance metric: {}", line);
        }
    }
}

/// Example of creating custom benchmark scenarios
#[allow(dead_code)]
fn custom_benchmark_scenarios() -> Result<(), Box<dyn std::error::Error>> {
    // Scenario 1: Memory-intensive operations benchmark
    let memory_config = BenchmarkConfig {
        warmup_iterations: 3,
        measurement_iterations: 10,
        measure_memory: true,
        calculate_flops: false,
        min_execution_time: Duration::from_micros(100),
        max_execution_time: Duration::from_secs(60),
    };

    let _memory_suite = BenchmarkSuite::new(memory_config);

    // Scenario 2: Compute-intensive operations benchmark
    let compute_config = BenchmarkConfig {
        warmup_iterations: 5,
        measurement_iterations: 15,
        measure_memory: false,
        calculate_flops: true,
        min_execution_time: Duration::from_micros(10),
        max_execution_time: Duration::from_secs(90),
    };

    let _compute_suite = BenchmarkSuite::new(compute_config);

    Ok(())
}
