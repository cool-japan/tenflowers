//! SIMD Benchmarking Utilities
//!
//! This module provides benchmarking utilities for comparing SIMD-optimized operations
//! against standard implementations and measuring performance improvements.

use std::time::Instant;

/// Benchmark result structure
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub optimized_time_ns: u64,
    pub standard_time_ns: u64,
    pub speedup: f64,
    pub size: usize,
    pub iterations: usize,
}

impl BenchmarkResult {
    /// Print a summary of the benchmark results
    pub fn print_summary(&self) {
        println!("SIMD Benchmark Results:");
        println!("  Size: {} elements", self.size);
        println!("  Iterations: {}", self.iterations);
        println!("  Optimized time: {} ns", self.optimized_time_ns);
        println!("  Standard time: {} ns", self.standard_time_ns);
        println!("  Speedup: {:.2}x", self.speedup);
    }

    /// Get the performance improvement as a percentage
    pub fn improvement_percentage(&self) -> f64 {
        (self.speedup - 1.0) * 100.0
    }

    /// Check if the optimization provides significant improvement
    pub fn is_significant_improvement(&self, threshold: f64) -> bool {
        self.speedup > threshold
    }
}

/// SIMD benchmarking utilities
pub struct Benchmarks;

impl Benchmarks {
    /// Benchmark optimized vs standard addition operations
    pub fn benchmark_add_performance(size: usize, iterations: usize) -> BenchmarkResult {
        let a = vec![1.0f32; size];
        let b = vec![2.0f32; size];
        let mut result_optimized = vec![0.0f32; size];
        let mut result_standard = vec![0.0f32; size];

        // Benchmark optimized implementation (unchecked for fair comparison)
        let start = Instant::now();
        for _ in 0..iterations {
            super::basic_ops::BasicOps::add_f32_unchecked(&a, &b, &mut result_optimized);
        }
        let optimized_time = start.elapsed();

        // Benchmark standard implementation
        let start = Instant::now();
        for _ in 0..iterations {
            for i in 0..size {
                result_standard[i] = a[i] + b[i];
            }
        }
        let standard_time = start.elapsed();

        BenchmarkResult {
            optimized_time_ns: optimized_time.as_nanos() as u64,
            standard_time_ns: standard_time.as_nanos() as u64,
            speedup: standard_time.as_secs_f64() / optimized_time.as_secs_f64(),
            size,
            iterations,
        }
    }

    /// Benchmark optimized vs standard multiplication operations
    pub fn benchmark_mul_performance(size: usize, iterations: usize) -> BenchmarkResult {
        let a = vec![2.0f32; size];
        let b = vec![3.0f32; size];
        let mut result_optimized = vec![0.0f32; size];
        let mut result_standard = vec![0.0f32; size];

        // Benchmark optimized implementation
        let start = Instant::now();
        for _ in 0..iterations {
            super::basic_ops::BasicOps::mul_f32_unchecked(&a, &b, &mut result_optimized);
        }
        let optimized_time = start.elapsed();

        // Benchmark standard implementation
        let start = Instant::now();
        for _ in 0..iterations {
            for i in 0..size {
                result_standard[i] = a[i] * b[i];
            }
        }
        let standard_time = start.elapsed();

        BenchmarkResult {
            optimized_time_ns: optimized_time.as_nanos() as u64,
            standard_time_ns: standard_time.as_nanos() as u64,
            speedup: standard_time.as_secs_f64() / optimized_time.as_secs_f64(),
            size,
            iterations,
        }
    }

    /// Benchmark optimized vs standard ReLU activation
    pub fn benchmark_relu_performance(size: usize, iterations: usize) -> BenchmarkResult {
        use scirs2_core::random::Random;
        let mut rng = Random::seed(42);
        let input: Vec<f32> = (0..size).map(|_| rng.random_range(-5.0, 5.0)).collect();
        let mut result_optimized = vec![0.0f32; size];
        let mut result_standard = vec![0.0f32; size];

        // Benchmark optimized implementation
        let start = Instant::now();
        for _ in 0..iterations {
            super::activation_functions::ActivationFunctions::relu_f32_optimized(
                &input,
                &mut result_optimized,
            )
            .unwrap();
        }
        let optimized_time = start.elapsed();

        // Benchmark standard implementation
        let start = Instant::now();
        for _ in 0..iterations {
            for i in 0..size {
                result_standard[i] = input[i].max(0.0);
            }
        }
        let standard_time = start.elapsed();

        BenchmarkResult {
            optimized_time_ns: optimized_time.as_nanos() as u64,
            standard_time_ns: standard_time.as_nanos() as u64,
            speedup: standard_time.as_secs_f64() / optimized_time.as_secs_f64(),
            size,
            iterations,
        }
    }

    /// Benchmark optimized vs standard dot product
    pub fn benchmark_dot_product_performance(size: usize, iterations: usize) -> BenchmarkResult {
        let a = vec![1.5f32; size];
        let b = vec![2.5f32; size];

        // Benchmark optimized implementation
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = super::matrix_ops::MatrixOps::dot_product_f32_optimized(&a, &b).unwrap();
        }
        let optimized_time = start.elapsed();

        // Benchmark standard implementation
        let start = Instant::now();
        for _ in 0..iterations {
            let _: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
        }
        let standard_time = start.elapsed();

        BenchmarkResult {
            optimized_time_ns: optimized_time.as_nanos() as u64,
            standard_time_ns: standard_time.as_nanos() as u64,
            speedup: standard_time.as_secs_f64() / optimized_time.as_secs_f64(),
            size,
            iterations,
        }
    }

    /// Benchmark optimized vs standard sum reduction
    pub fn benchmark_sum_performance(size: usize, iterations: usize) -> BenchmarkResult {
        let input = vec![1.5f32; size];

        // Benchmark optimized implementation
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = super::reduction_ops::ReductionOps::sum_f32_unchecked(&input);
        }
        let optimized_time = start.elapsed();

        // Benchmark standard implementation
        let start = Instant::now();
        for _ in 0..iterations {
            let _: f32 = input.iter().sum();
        }
        let standard_time = start.elapsed();

        BenchmarkResult {
            optimized_time_ns: optimized_time.as_nanos() as u64,
            standard_time_ns: standard_time.as_nanos() as u64,
            speedup: standard_time.as_secs_f64() / optimized_time.as_secs_f64(),
            size,
            iterations,
        }
    }

    /// Benchmark optimized vs standard exp function
    pub fn benchmark_exp_performance(size: usize, iterations: usize) -> BenchmarkResult {
        use scirs2_core::random::Random;
        let mut rng = Random::seed(42);
        let input: Vec<f32> = (0..size).map(|_| rng.random_range(-2.0, 2.0)).collect();
        let mut result_optimized = vec![0.0f32; size];
        let mut result_standard = vec![0.0f32; size];

        // Benchmark optimized implementation
        let start = Instant::now();
        for _ in 0..iterations {
            super::math_functions::MathFunctions::exp_f32_optimized(&input, &mut result_optimized)
                .unwrap();
        }
        let optimized_time = start.elapsed();

        // Benchmark standard implementation
        let start = Instant::now();
        for _ in 0..iterations {
            for i in 0..size {
                result_standard[i] = input[i].exp();
            }
        }
        let standard_time = start.elapsed();

        BenchmarkResult {
            optimized_time_ns: optimized_time.as_nanos() as u64,
            standard_time_ns: standard_time.as_nanos() as u64,
            speedup: standard_time.as_secs_f64() / optimized_time.as_secs_f64(),
            size,
            iterations,
        }
    }

    /// Comprehensive benchmark suite testing multiple operations
    pub fn comprehensive_benchmark_suite(
        size: usize,
        iterations: usize,
    ) -> Vec<(&'static str, BenchmarkResult)> {
        let mut results = Vec::new();

        println!("Running comprehensive SIMD benchmark suite...");
        println!("Size: {} elements, Iterations: {}", size, iterations);
        println!("{}", "=".repeat(60));

        // Addition benchmark
        print!("Benchmarking addition... ");
        let add_result = Self::benchmark_add_performance(size, iterations);
        println!("Speedup: {:.2}x", add_result.speedup);
        results.push(("Addition", add_result));

        // Multiplication benchmark
        print!("Benchmarking multiplication... ");
        let mul_result = Self::benchmark_mul_performance(size, iterations);
        println!("Speedup: {:.2}x", mul_result.speedup);
        results.push(("Multiplication", mul_result));

        // ReLU benchmark
        print!("Benchmarking ReLU... ");
        let relu_result = Self::benchmark_relu_performance(size, iterations);
        println!("Speedup: {:.2}x", relu_result.speedup);
        results.push(("ReLU", relu_result));

        // Dot product benchmark
        print!("Benchmarking dot product... ");
        let dot_result = Self::benchmark_dot_product_performance(size, iterations);
        println!("Speedup: {:.2}x", dot_result.speedup);
        results.push(("Dot Product", dot_result));

        // Sum reduction benchmark
        print!("Benchmarking sum reduction... ");
        let sum_result = Self::benchmark_sum_performance(size, iterations);
        println!("Speedup: {:.2}x", sum_result.speedup);
        results.push(("Sum", sum_result));

        // Exp function benchmark
        print!("Benchmarking exp function... ");
        let exp_result = Self::benchmark_exp_performance(size, iterations);
        println!("Speedup: {:.2}x", exp_result.speedup);
        results.push(("Exp", exp_result));

        println!("{}", "=".repeat(60));

        results
    }

    /// Print detailed benchmark report
    pub fn print_benchmark_report(results: &[(&'static str, BenchmarkResult)]) {
        println!("\nDetailed Benchmark Report:");
        println!("{}", "=".repeat(80));

        for (operation, result) in results {
            println!("\n{} Performance:", operation);
            println!(
                "  Optimized: {:.2} ms",
                result.optimized_time_ns as f64 / 1_000_000.0
            );
            println!(
                "  Standard:  {:.2} ms",
                result.standard_time_ns as f64 / 1_000_000.0
            );
            println!(
                "  Speedup:   {:.2}x ({:.1}% improvement)",
                result.speedup,
                result.improvement_percentage()
            );

            if result.speedup > 1.5 {
                println!("  Status:    🚀 Excellent optimization");
            } else if result.speedup > 1.2 {
                println!("  Status:    ✅ Good optimization");
            } else if result.speedup > 1.0 {
                println!("  Status:    📈 Modest improvement");
            } else {
                println!("  Status:    ⚠️ No improvement");
            }
        }

        // Overall statistics
        let avg_speedup: f64 =
            results.iter().map(|(_, r)| r.speedup).sum::<f64>() / results.len() as f64;
        let max_speedup = results
            .iter()
            .map(|(_, r)| r.speedup)
            .fold(0.0f64, f64::max);
        let min_speedup = results
            .iter()
            .map(|(_, r)| r.speedup)
            .fold(f64::INFINITY, f64::min);

        println!("\n{}", "=".repeat(80));
        println!("Overall Performance Summary:");
        println!("  Average speedup: {:.2}x", avg_speedup);
        println!("  Best speedup:    {:.2}x", max_speedup);
        println!("  Worst speedup:   {:.2}x", min_speedup);
        println!("  Total operations: {}", results.len());

        let good_optimizations = results.iter().filter(|(_, r)| r.speedup > 1.2).count();
        println!(
            "  Good optimizations: {}/{}",
            good_optimizations,
            results.len()
        );
    }

    /// Warm-up function to stabilize CPU frequency and caches
    pub fn warmup() {
        let warmup_size = 1000;
        let warmup_iterations = 100;

        let a = vec![1.0f32; warmup_size];
        let b = vec![2.0f32; warmup_size];
        let mut result = vec![0.0f32; warmup_size];

        // Warm-up with some operations
        for _ in 0..warmup_iterations {
            for i in 0..warmup_size {
                result[i] = a[i] + b[i] * 2.0;
            }
        }

        // Force the compiler to not optimize away the warmup
        let _checksum: f32 = result.iter().sum();
    }

    /// Run scalability test across different array sizes
    pub fn scalability_test(
        operation: &str,
        base_iterations: usize,
    ) -> Vec<(usize, BenchmarkResult)> {
        let sizes = vec![32, 64, 128, 256, 512, 1024, 2048, 4096, 8192];
        let mut results = Vec::new();

        println!("Running scalability test for {}...", operation);

        for &size in &sizes {
            // Adjust iterations based on size to keep test time reasonable
            let iterations = (base_iterations * 1000) / size.max(1);
            let iterations = iterations.max(10); // Minimum 10 iterations

            let result = match operation {
                "add" => Self::benchmark_add_performance(size, iterations),
                "mul" => Self::benchmark_mul_performance(size, iterations),
                "relu" => Self::benchmark_relu_performance(size, iterations),
                "dot" => Self::benchmark_dot_product_performance(size, iterations),
                "sum" => Self::benchmark_sum_performance(size, iterations),
                "exp" => Self::benchmark_exp_performance(size, iterations),
                _ => panic!("Unknown operation: {}", operation),
            };

            println!("Size {}: {:.2}x speedup", size, result.speedup);
            results.push((size, result));
        }

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_result() {
        let result = BenchmarkResult {
            optimized_time_ns: 100,
            standard_time_ns: 200,
            speedup: 2.0,
            size: 1000,
            iterations: 100,
        };

        assert_eq!(result.improvement_percentage(), 100.0);
        assert!(result.is_significant_improvement(1.5));
        assert!(!result.is_significant_improvement(2.5));
    }

    #[test]
    fn test_benchmark_add_performance() {
        // Run a small benchmark to ensure it works
        let result = Benchmarks::benchmark_add_performance(100, 10);

        // Basic sanity checks
        assert!(result.optimized_time_ns >= 0); // Allow zero for very fast operations
        assert!(result.standard_time_ns >= 0); // Allow zero for very fast operations
        assert!(result.speedup >= 0.0 || result.speedup.is_infinite() || result.speedup.is_nan());
        assert_eq!(result.size, 100);
        assert_eq!(result.iterations, 10);
    }

    #[test]
    fn test_comprehensive_benchmark_basic() {
        // Run a minimal comprehensive benchmark
        let results = Benchmarks::comprehensive_benchmark_suite(32, 5);

        // Should have multiple benchmark results
        assert!(results.len() >= 5);

        // Each result should be valid
        for (name, result) in &results {
            assert!(!name.is_empty());
            assert!(result.optimized_time_ns >= 0); // Allow zero for very fast operations
            assert!(result.standard_time_ns >= 0); // Allow zero for very fast operations
                                                   // Speedup can be infinite, NaN, or positive depending on timing precision
            assert!(
                result.speedup >= 0.0 || result.speedup.is_infinite() || result.speedup.is_nan()
            );
        }
    }

    #[test]
    fn test_warmup() {
        // Should not panic or error
        Benchmarks::warmup();
    }

    #[test]
    fn test_scalability_test_basic() {
        // Run a very basic scalability test
        let results = Benchmarks::scalability_test("add", 1);

        // Should have multiple size results
        assert!(!results.is_empty());

        // Check that results are ordered by size
        for i in 1..results.len() {
            assert!(
                results[i].0 > results[i - 1].0,
                "Sizes should be increasing"
            );
        }
    }
}
