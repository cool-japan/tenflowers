// Benchmark Demonstration for TenfloweRS
// Shows the sophisticated benchmarking capabilities

use tenflowers_core::{
    run_simple_benchmarks, validate_optimizations, SimpleBenchmarkSuite, SimpleBenchmarkConfig,
    Tensor, Result,
};

fn main() -> Result<()> {
    println!("🎯 === TENFLOWERS BENCHMARK DEMONSTRATION ===");
    println!("Demonstrating advanced benchmarking capabilities\n");

    // Run the simple benchmarks
    println!("🚀 Running TenfloweRS Benchmarks...");
    let report = run_simple_benchmarks()?;
    report.print_report();

    // Run optimization validation
    println!("\n🔍 Running Optimization Validation...");
    validate_optimizations()?;

    // Custom benchmark configuration demonstration
    println!("\n⚙️  Custom Benchmark Configuration Demo...");
    demonstrate_custom_benchmarks()?;

    println!("\n✅ Benchmark demonstration complete!");
    Ok(())
}

fn demonstrate_custom_benchmarks() -> Result<()> {
    let config = SimpleBenchmarkConfig {
        warmup_iterations: 3,
        benchmark_iterations: 10,
        test_sizes: vec![
            vec![64, 64],       // Very small for quick demo
            vec![256, 256],     // Small matrices
            vec![512, 512],     // Medium matrices
        ],
    };

    let mut suite = SimpleBenchmarkSuite::new(config);
    let report = suite.run_benchmarks()?;

    println!("  📊 Custom Configuration Results:");
    println!("    Performance Score: {:.2}/10", report.summary.performance_score);

    if report.summary.performance_score >= 5.0 {
        println!("    ✅ Performance targets met!");
    }

    Ok(())
}