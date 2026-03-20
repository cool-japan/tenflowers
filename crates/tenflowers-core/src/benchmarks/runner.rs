use super::{BenchmarkConfig, ComprehensiveBenchmarkSuite};
use crate::{Result, TensorError};
use std::time::Duration;

/// Main benchmark runner for TenfloweRS operations
pub struct BenchmarkRunner {
    suite: ComprehensiveBenchmarkSuite,
}

impl BenchmarkRunner {
    /// Create a new benchmark runner with default configuration
    pub fn new() -> Self {
        let config = BenchmarkConfig::default();
        Self::with_config(config)
    }

    /// Create a new benchmark runner with custom configuration
    pub fn with_config(config: BenchmarkConfig) -> Self {
        Self {
            suite: ComprehensiveBenchmarkSuite::new(config),
        }
    }

    /// Create a benchmark runner optimized for quick testing
    pub fn quick_test() -> Self {
        let config = BenchmarkConfig {
            warmup_iterations: 1,
            benchmark_iterations: 3,
            max_duration: Duration::from_secs(10),
            include_gpu: true,
            min_execution_time: Duration::from_micros(1),
        };
        Self::with_config(config)
    }

    /// Create a benchmark runner optimized for comprehensive testing
    pub fn comprehensive() -> Self {
        let config = BenchmarkConfig {
            warmup_iterations: 5,
            benchmark_iterations: 20,
            max_duration: Duration::from_secs(120),
            include_gpu: true,
            min_execution_time: Duration::from_micros(1),
        };
        Self::with_config(config)
    }

    /// Run all benchmarks and return a comprehensive report
    pub async fn run_all_benchmarks(&self) -> Result<String> {
        self.suite.run_complete_benchmark_suite().await
    }

    /// Run CPU vs GPU comparison benchmarks
    pub async fn run_cpu_gpu_comparison(&self) -> Result<String> {
        self.suite.generate_cpu_gpu_comparison().await
    }

    /// Run only GPU manipulation operation benchmarks
    pub async fn run_manipulation_benchmarks(&self) -> Result<String> {
        self.suite.manipulation_benchmarks.generate_manipulation_report().await
    }

    /// Run only convolution operation benchmarks
    pub async fn run_convolution_benchmarks(&self) -> Result<String> {
        self.suite.convolution_benchmarks.generate_convolution_report().await
    }

    /// Save benchmark results to a file
    pub async fn save_benchmark_report(&self, file_path: &str) -> Result<()> {
        let report = self.run_all_benchmarks().await?;
        std::fs::write(file_path, report).map_err(|e| {
            TensorError::io_error_simple(format!("Failed to write benchmark report to {}: {}", file_path, e))
        })?;
        Ok(())
    }

    /// Generate and print a quick benchmark summary
    pub async fn print_quick_summary(&self) -> Result<()> {
        println!("Running TenfloweRS benchmark suite...\n");
        
        // Run CPU vs GPU comparison for quick overview
        match self.run_cpu_gpu_comparison().await {
            Ok(report) => {
                println!("{}", report);
            }
            Err(e) => {
                println!("Error running benchmarks: {}", e);
            }
        }
        
        Ok(())
    }
}

impl Default for BenchmarkRunner {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience functions for running common benchmark scenarios
pub mod presets {
    use super::*;

    /// Run quick benchmarks for development testing
    pub async fn run_quick_benchmarks() -> Result<String> {
        let runner = BenchmarkRunner::quick_test();
        runner.run_all_benchmarks().await
    }

    /// Run comprehensive benchmarks for performance analysis
    pub async fn run_comprehensive_benchmarks() -> Result<String> {
        let runner = BenchmarkRunner::comprehensive();
        runner.run_all_benchmarks().await
    }

    /// Run and save benchmark results to file
    pub async fn benchmark_and_save(output_file: &str) -> Result<()> {
        let runner = BenchmarkRunner::new();
        runner.save_benchmark_report(output_file).await
    }

    /// Print a quick performance summary to console
    pub async fn print_performance_summary() -> Result<()> {
        let runner = BenchmarkRunner::quick_test();
        runner.print_quick_summary().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_runner_creation() {
        let runner = BenchmarkRunner::new();
        assert!(std::mem::size_of_val(&runner) > 0);
    }

    #[test]
    fn test_quick_test_config() {
        let runner = BenchmarkRunner::quick_test();
        assert!(std::mem::size_of_val(&runner) > 0);
    }

    #[test]
    fn test_comprehensive_config() {
        let runner = BenchmarkRunner::comprehensive();
        assert!(std::mem::size_of_val(&runner) > 0);
    }
}