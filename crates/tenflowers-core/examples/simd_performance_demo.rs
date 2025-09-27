//! SIMD Performance Optimization Demo
//!
//! This example demonstrates the performance improvements achieved through
//! SIMD-optimized tensor operations in TenfloweRS.

#[cfg(feature = "simd")]
use tenflowers_core::{simd_benchmarks, SimdCapabilities, SimdOptimizer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 TenfloweRS SIMD Performance Optimization Demo");
    println!("================================================\n");

    #[cfg(feature = "simd")]
    {
        // Detect SIMD capabilities
        let capabilities = SimdOptimizer::detect_capabilities();
        print_capabilities(&capabilities);

        // Run performance benchmarks
        run_performance_benchmarks();

        // Demonstrate different optimized operations
        demonstrate_operations()?;
    }

    #[cfg(not(feature = "simd"))]
    {
        println!(
            "❌ SIMD optimizations not enabled. Compile with --features simd to see the demo."
        );
        println!("\nTo enable SIMD optimizations, run:");
        println!("   cargo run --example simd_performance_demo --features simd");
    }

    Ok(())
}

#[cfg(feature = "simd")]
fn print_capabilities(caps: &SimdCapabilities) {
    println!("🔍 Platform SIMD Capabilities:");
    println!(
        "  Auto-vectorization: {}",
        if caps.has_auto_vectorization {
            "✅"
        } else {
            "❌"
        }
    );
    println!(
        "  Target features: {}",
        if caps.has_target_features {
            "✅"
        } else {
            "❌"
        }
    );
    println!(
        "  Recommended unroll factor: {}",
        caps.recommended_unroll_factor
    );
    println!("  Cache line size: {} bytes\n", caps.cache_line_size);
}

#[cfg(feature = "simd")]
fn run_performance_benchmarks() {
    println!("⚡ Performance Benchmarks:");

    let sizes = vec![1_000, 10_000, 100_000];
    let iterations = 1_000;

    for size in sizes {
        let result = simd_benchmarks::benchmark_add_performance(size, iterations);

        println!("  Size: {} elements", size);
        println!("    Optimized: {} ns", result.optimized_time_ns);
        println!("    Standard:  {} ns", result.standard_time_ns);
        println!("    Speedup:   {:.2}x", result.speedup);

        let improvement = if result.speedup > 1.0 {
            format!("🏎️  {:.0}% faster", (result.speedup - 1.0) * 100.0)
        } else {
            format!("⚖️  Similar performance ({:.2}x)", result.speedup)
        };
        println!("    Result:    {}\n", improvement);
    }
}

#[cfg(feature = "simd")]
fn demonstrate_operations() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧮 SIMD Operation Demonstrations:");

    // Element-wise addition
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let mut result = vec![0.0; 8];

    SimdOptimizer::add_f32_optimized(&a, &b, &mut result)?;
    println!("  Addition: {:?} + {:?} = {:?}", a, b, result);

    // Element-wise multiplication
    let mut mul_result = vec![0.0; 8];
    SimdOptimizer::mul_f32_optimized(&a, &b, &mut mul_result)?;
    println!("  Multiplication: {:?} * {:?} = {:?}", a, b, mul_result);

    // ReLU activation
    let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, -4.0, 5.0];
    let mut relu_output = vec![0.0; 8];
    SimdOptimizer::relu_f32_optimized(&input, &mut relu_output)?;
    println!("  ReLU: {:?} -> {:?}", input, relu_output);

    // Dot product
    let vec1 = vec![1.0, 2.0, 3.0, 4.0];
    let vec2 = vec![2.0, 3.0, 4.0, 5.0];
    let dot_result = SimdOptimizer::dot_product_f32_optimized(&vec1, &vec2)?;
    println!("  Dot Product: {:?} · {:?} = {}", vec1, vec2, dot_result);

    // Auto-selection demonstration
    let small_a = vec![1.0, 2.0, 3.0];
    let small_b = vec![4.0, 5.0, 6.0];
    let mut auto_result = vec![0.0; 3];
    SimdOptimizer::add_f32_auto(&small_a, &small_b, &mut auto_result)?;
    println!(
        "  Auto-selection (small): {:?} + {:?} = {:?}",
        small_a, small_b, auto_result
    );

    println!("\n✨ All SIMD operations completed successfully!");

    Ok(())
}

#[cfg(feature = "simd")]
mod benchmarking_utilities {
    use super::*;

    /// Run comprehensive benchmarks across different array sizes
    pub fn comprehensive_benchmark() {
        println!("📊 Comprehensive SIMD Performance Analysis:");

        let test_cases = vec![
            ("Small arrays", 100, 10_000),
            ("Medium arrays", 10_000, 1_000),
            ("Large arrays", 1_000_000, 100),
        ];

        for (name, size, iterations) in test_cases {
            println!("\n{} ({} elements, {} iterations):", name, size, iterations);

            let result = simd_benchmarks::benchmark_add_performance(size, iterations);
            result.print_summary();

            // Performance classification
            match result.speedup {
                x if x >= 2.0 => println!("   🚀 Excellent optimization!"),
                x if x >= 1.5 => println!("   ⚡ Good optimization"),
                x if x >= 1.1 => println!("   📈 Moderate improvement"),
                _ => println!("   ⚖️ Similar performance"),
            }
        }
    }
}

// Run comprehensive benchmarks if compiled as a standalone binary
#[cfg(all(feature = "simd", not(test)))]
#[allow(dead_code)]
fn extended_demo() {
    benchmarking_utilities::comprehensive_benchmark();
}
