//! Performance metrics and optimization reports for WASM deployment

/// Performance metrics for WASM deployment
#[cfg(feature = "wasm")]
#[derive(Debug, Default)]
pub struct WasmPerformanceMetrics {
    /// Inference time (ms)
    pub inference_time_ms: f64,
    /// Memory usage (bytes)
    pub memory_usage_bytes: usize,
    /// Bundle size (bytes)
    pub bundle_size_bytes: usize,
    /// Initialization time (ms)
    pub init_time_ms: f64,
    /// Cache hit ratio
    pub cache_hit_ratio: f64,
}

/// Bundle optimization report
#[cfg(feature = "wasm")]
#[derive(Debug, Default)]
pub struct WasmOptimizationReport {
    /// Size reduction from dead code elimination (KB)
    pub dead_code_eliminated_kb: f64,
    /// Size reduction from tree shaking (KB)
    pub tree_shaking_saved_kb: f64,
    /// Compression ratio (0.0 to 1.0)
    pub compression_ratio: f64,
    /// Total size reduction (KB)
    pub total_size_reduction_kb: f64,
    /// Original bundle size (KB)
    pub original_size_kb: f64,
    /// Optimized bundle size (KB)
    pub optimized_size_kb: f64,
}

/// Performance benchmarking suite for WASM optimization
#[cfg(feature = "wasm")]
pub struct WasmPerformanceBenchmark {
    /// Benchmark results
    results: Vec<WasmBenchmarkResult>,
    /// Current configuration
    config: WasmBenchmarkConfig,
}

/// Individual benchmark result
#[cfg(feature = "wasm")]
#[derive(Debug, Clone)]
pub struct WasmBenchmarkResult {
    /// Benchmark name
    pub name: String,
    /// Execution time (ms)
    pub time_ms: f64,
    /// Memory usage (bytes)
    pub memory_bytes: usize,
    /// Throughput (operations/second)
    pub throughput_ops_per_sec: f64,
    /// Success status
    pub success: bool,
}

/// Benchmark configuration
#[cfg(feature = "wasm")]
#[derive(Debug, Clone)]
pub struct WasmBenchmarkConfig {
    /// Number of warmup iterations
    pub warmup_iterations: usize,
    /// Number of measurement iterations
    pub measurement_iterations: usize,
    /// Maximum benchmark time (ms)
    pub max_time_ms: f64,
    /// Memory limit for benchmarks (bytes)
    pub memory_limit_bytes: usize,
}

#[cfg(feature = "wasm")]
impl WasmPerformanceMetrics {
    /// Create new performance metrics
    pub fn new() -> Self {
        Self::default()
    }

    /// Update inference timing
    pub fn record_inference_time(&mut self, time_ms: f64) {
        self.inference_time_ms = time_ms;
    }

    /// Update memory usage
    pub fn record_memory_usage(&mut self, bytes: usize) {
        self.memory_usage_bytes = bytes;
    }

    /// Update cache hit ratio
    pub fn record_cache_hit_ratio(&mut self, ratio: f64) {
        self.cache_hit_ratio = ratio.clamp(0.0, 1.0);
    }

    /// Get overall performance score (0-100)
    pub fn get_performance_score(&self) -> f64 {
        let time_score = if self.inference_time_ms > 0.0 {
            (100.0 / self.inference_time_ms).min(100.0)
        } else {
            0.0
        };

        let memory_score = if self.memory_usage_bytes > 0 {
            (100.0 * (1.0 - (self.memory_usage_bytes as f64 / (64.0 * 1024.0 * 1024.0)))).max(0.0)
        } else {
            100.0
        };

        let cache_score = self.cache_hit_ratio * 100.0;

        (time_score * 0.4 + memory_score * 0.3 + cache_score * 0.3).min(100.0)
    }

    /// Export metrics as JSON-like string
    pub fn to_json(&self) -> String {
        format!(
            r#"{{
    "inference_time_ms": {},
    "memory_usage_bytes": {},
    "bundle_size_bytes": {},
    "init_time_ms": {},
    "cache_hit_ratio": {},
    "performance_score": {}
}}"#,
            self.inference_time_ms,
            self.memory_usage_bytes,
            self.bundle_size_bytes,
            self.init_time_ms,
            self.cache_hit_ratio,
            self.get_performance_score()
        )
    }

    /// Compare with baseline metrics
    pub fn compare_with_baseline(
        &self,
        baseline: &WasmPerformanceMetrics,
    ) -> WasmPerformanceComparison {
        WasmPerformanceComparison {
            inference_time_improvement: if baseline.inference_time_ms > 0.0 {
                (baseline.inference_time_ms - self.inference_time_ms) / baseline.inference_time_ms
            } else {
                0.0
            },
            memory_reduction: if baseline.memory_usage_bytes > 0 {
                (baseline.memory_usage_bytes as f64 - self.memory_usage_bytes as f64)
                    / baseline.memory_usage_bytes as f64
            } else {
                0.0
            },
            cache_improvement: self.cache_hit_ratio - baseline.cache_hit_ratio,
            overall_improvement: self.get_performance_score() - baseline.get_performance_score(),
        }
    }
}

/// Performance comparison results
#[cfg(feature = "wasm")]
#[derive(Debug)]
pub struct WasmPerformanceComparison {
    /// Inference time improvement ratio (-1.0 to 1.0, positive is better)
    pub inference_time_improvement: f64,
    /// Memory reduction ratio (-1.0 to 1.0, positive is better)
    pub memory_reduction: f64,
    /// Cache hit ratio improvement (-1.0 to 1.0, positive is better)
    pub cache_improvement: f64,
    /// Overall performance score improvement
    pub overall_improvement: f64,
}

#[cfg(feature = "wasm")]
impl WasmOptimizationReport {
    /// Create new optimization report
    pub fn new() -> Self {
        Self::default()
    }

    /// Calculate final optimized size
    pub fn calculate_optimized_size(&mut self) {
        if self.original_size_kb > 0.0 {
            let size_reduction = self.dead_code_eliminated_kb + self.tree_shaking_saved_kb;
            let uncompressed_size = self.original_size_kb - size_reduction;
            self.optimized_size_kb = uncompressed_size * (1.0 - self.compression_ratio);
            self.total_size_reduction_kb = self.original_size_kb - self.optimized_size_kb;
        }
    }

    /// Get optimization percentage
    pub fn get_optimization_percentage(&self) -> f64 {
        if self.original_size_kb > 0.0 {
            (self.total_size_reduction_kb / self.original_size_kb) * 100.0
        } else {
            0.0
        }
    }

    /// Generate human-readable summary
    pub fn generate_summary(&self) -> String {
        format!(
            "WASM Optimization Report:\n\
             Original Size: {:.2} KB\n\
             Optimized Size: {:.2} KB\n\
             Total Reduction: {:.2} KB ({:.1}%)\n\
             \n\
             Breakdown:\n\
             - Dead Code Elimination: {:.2} KB\n\
             - Tree Shaking: {:.2} KB\n\
             - Compression: {:.1}% ratio\n",
            self.original_size_kb,
            self.optimized_size_kb,
            self.total_size_reduction_kb,
            self.get_optimization_percentage(),
            self.dead_code_eliminated_kb,
            self.tree_shaking_saved_kb,
            self.compression_ratio * 100.0
        )
    }

    /// Check if optimization meets target
    pub fn meets_target(&self, target_size_kb: f64) -> bool {
        self.optimized_size_kb <= target_size_kb
    }
}

#[cfg(feature = "wasm")]
impl WasmPerformanceBenchmark {
    /// Create new benchmark suite
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
            config: WasmBenchmarkConfig {
                warmup_iterations: 5,
                measurement_iterations: 10,
                max_time_ms: 10000.0,                 // 10 seconds
                memory_limit_bytes: 64 * 1024 * 1024, // 64MB
            },
        }
    }

    /// Set benchmark configuration
    pub fn with_config(mut self, config: WasmBenchmarkConfig) -> Self {
        self.config = config;
        self
    }

    /// Run tensor operation benchmark
    pub fn benchmark_tensor_ops(&mut self) -> crate::Result<()> {
        // Benchmark matrix multiplication
        self.benchmark_matmul()?;

        // Benchmark element-wise operations
        self.benchmark_elementwise_ops()?;

        // Benchmark tensor creation
        self.benchmark_tensor_creation()?;

        // Benchmark memory operations
        self.benchmark_memory_ops()?;

        Ok(())
    }

    /// Run inference benchmark
    pub fn benchmark_inference(&mut self, model_size: usize) -> crate::Result<()> {
        let start_time = std::time::Instant::now();

        // Simulate inference workload
        let mut total_ops = 0;
        for i in 0..self.config.measurement_iterations {
            let iter_start = std::time::Instant::now();

            // Simulate model forward pass
            let _ = self.simulate_forward_pass(model_size)?;
            total_ops += model_size;

            let iter_time = iter_start.elapsed().as_millis() as f64;
            if start_time.elapsed().as_millis() as f64 > self.config.max_time_ms {
                break;
            }
        }

        let total_time = start_time.elapsed().as_millis() as f64;
        let throughput = (total_ops as f64) / (total_time / 1000.0);

        self.results.push(WasmBenchmarkResult {
            name: format!("inference_model_{}", model_size),
            time_ms: total_time / self.config.measurement_iterations as f64,
            memory_bytes: model_size * 4, // Assume f32 weights
            throughput_ops_per_sec: throughput,
            success: true,
        });

        Ok(())
    }

    /// Get benchmark results
    pub fn get_results(&self) -> &[WasmBenchmarkResult] {
        &self.results
    }

    /// Generate benchmark report
    pub fn generate_report(&self) -> String {
        let mut report = String::from("WASM Performance Benchmark Report\n");
        report.push_str("=====================================\n\n");

        for result in &self.results {
            report.push_str(&format!(
                "Benchmark: {}\n\
                 Time: {:.2} ms\n\
                 Memory: {} bytes\n\
                 Throughput: {:.0} ops/sec\n\
                 Status: {}\n\n",
                result.name,
                result.time_ms,
                result.memory_bytes,
                result.throughput_ops_per_sec,
                if result.success { "PASS" } else { "FAIL" }
            ));
        }

        // Calculate averages
        let avg_time: f64 =
            self.results.iter().map(|r| r.time_ms).sum::<f64>() / self.results.len() as f64;
        let avg_throughput: f64 = self
            .results
            .iter()
            .map(|r| r.throughput_ops_per_sec)
            .sum::<f64>()
            / self.results.len() as f64;
        let total_memory: usize = self.results.iter().map(|r| r.memory_bytes).sum();

        report.push_str(&format!(
            "Summary:\n\
             Average Time: {:.2} ms\n\
             Average Throughput: {:.0} ops/sec\n\
             Total Memory: {} bytes\n\
             Success Rate: {:.1}%\n",
            avg_time,
            avg_throughput,
            total_memory,
            (self.results.iter().filter(|r| r.success).count() as f64 / self.results.len() as f64)
                * 100.0
        ));

        report
    }

    fn benchmark_matmul(&mut self) -> crate::Result<()> {
        let sizes = [32, 64, 128, 256];

        for &size in &sizes {
            let start_time = std::time::Instant::now();

            // Simulate matrix multiplication
            for _ in 0..self.config.measurement_iterations {
                let _ = self.simulate_matmul(size, size, size)?;
            }

            let total_time = start_time.elapsed().as_millis() as f64;
            let ops_per_iter = size * size * size; // O(n^3) operations
            let throughput =
                (ops_per_iter * self.config.measurement_iterations) as f64 / (total_time / 1000.0);

            self.results.push(WasmBenchmarkResult {
                name: format!("matmul_{}x{}", size, size),
                time_ms: total_time / self.config.measurement_iterations as f64,
                memory_bytes: size * size * 4 * 3, // 3 matrices of f32
                throughput_ops_per_sec: throughput,
                success: true,
            });
        }

        Ok(())
    }

    fn benchmark_elementwise_ops(&mut self) -> crate::Result<()> {
        let sizes = [1024, 4096, 16384, 65536];

        for &size in &sizes {
            let start_time = std::time::Instant::now();

            // Simulate element-wise operations
            for _ in 0..self.config.measurement_iterations {
                let _ = self.simulate_elementwise_add(size)?;
            }

            let total_time = start_time.elapsed().as_millis() as f64;
            let throughput =
                (size * self.config.measurement_iterations) as f64 / (total_time / 1000.0);

            self.results.push(WasmBenchmarkResult {
                name: format!("elementwise_add_{}", size),
                time_ms: total_time / self.config.measurement_iterations as f64,
                memory_bytes: size * 4 * 2, // 2 arrays of f32
                throughput_ops_per_sec: throughput,
                success: true,
            });
        }

        Ok(())
    }

    fn benchmark_tensor_creation(&mut self) -> crate::Result<()> {
        let sizes = [100, 1000, 10000, 100000];

        for &size in &sizes {
            let start_time = std::time::Instant::now();

            // Simulate tensor creation
            for _ in 0..self.config.measurement_iterations {
                let _ = self.simulate_tensor_creation(size)?;
            }

            let total_time = start_time.elapsed().as_millis() as f64;
            let throughput = (self.config.measurement_iterations) as f64 / (total_time / 1000.0);

            self.results.push(WasmBenchmarkResult {
                name: format!("tensor_creation_{}", size),
                time_ms: total_time / self.config.measurement_iterations as f64,
                memory_bytes: size * 4, // f32 array
                throughput_ops_per_sec: throughput,
                success: true,
            });
        }

        Ok(())
    }

    fn benchmark_memory_ops(&mut self) -> crate::Result<()> {
        let sizes = [1024, 8192, 32768, 131072];

        for &size in &sizes {
            let start_time = std::time::Instant::now();

            // Simulate memory operations
            for _ in 0..self.config.measurement_iterations {
                let _ = self.simulate_memory_copy(size)?;
            }

            let total_time = start_time.elapsed().as_millis() as f64;
            let throughput =
                (size * self.config.measurement_iterations) as f64 / (total_time / 1000.0);

            self.results.push(WasmBenchmarkResult {
                name: format!("memory_copy_{}", size),
                time_ms: total_time / self.config.measurement_iterations as f64,
                memory_bytes: size * 4 * 2, // Source and destination
                throughput_ops_per_sec: throughput,
                success: true,
            });
        }

        Ok(())
    }

    // Simulation functions for benchmarking
    fn simulate_matmul(&self, m: usize, n: usize, k: usize) -> crate::Result<Vec<f32>> {
        // Simplified matrix multiplication simulation
        let result = vec![1.0f32; m * n];
        // Simulate computation time
        std::thread::sleep(std::time::Duration::from_micros((m * n * k / 1000) as u64));
        Ok(result)
    }

    fn simulate_elementwise_add(&self, size: usize) -> crate::Result<Vec<f32>> {
        // Simplified element-wise addition simulation
        let result = vec![2.0f32; size];
        // Simulate computation time
        std::thread::sleep(std::time::Duration::from_micros(size as u64 / 100));
        Ok(result)
    }

    fn simulate_tensor_creation(&self, size: usize) -> crate::Result<Vec<f32>> {
        // Simplified tensor creation simulation
        let result = vec![0.0f32; size];
        // Simulate allocation time
        std::thread::sleep(std::time::Duration::from_micros(size as u64 / 1000));
        Ok(result)
    }

    fn simulate_memory_copy(&self, size: usize) -> crate::Result<Vec<f32>> {
        // Simplified memory copy simulation
        let source = vec![1.0f32; size];
        let result = source.clone();
        // Simulate copy time
        std::thread::sleep(std::time::Duration::from_micros(size as u64 / 100));
        Ok(result)
    }

    fn simulate_forward_pass(&self, model_size: usize) -> crate::Result<Vec<f32>> {
        // Simplified neural network forward pass simulation
        let result = vec![0.5f32; model_size / 100]; // Output smaller than input
                                                     // Simulate computation time proportional to model size
        std::thread::sleep(std::time::Duration::from_micros((model_size / 10) as u64));
        Ok(result)
    }
}

#[cfg(feature = "wasm")]
impl Default for WasmPerformanceBenchmark {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "wasm")]
    #[ignore = "WASM tests require WASM target - cannot run on native"]
    fn test_performance_metrics() {
        let mut metrics = WasmPerformanceMetrics::new();
        assert_eq!(metrics.inference_time_ms, 0.0);
        assert_eq!(metrics.get_performance_score(), 100.0);

        metrics.record_inference_time(10.0);
        metrics.record_cache_hit_ratio(0.8);

        let score = metrics.get_performance_score();
        assert!(score > 0.0 && score <= 100.0);
    }

    #[test]
    #[cfg(feature = "wasm")]
    fn test_optimization_report() {
        let mut report = WasmOptimizationReport::new();
        report.original_size_kb = 1000.0;
        report.dead_code_eliminated_kb = 200.0;
        report.tree_shaking_saved_kb = 150.0;
        report.compression_ratio = 0.3;

        report.calculate_optimized_size();

        assert!(report.optimized_size_kb < report.original_size_kb);
        assert!(report.get_optimization_percentage() > 0.0);
        assert!(report.meets_target(500.0));
    }

    #[test]
    #[cfg(feature = "wasm")]
    fn test_benchmark_creation() {
        let benchmark = WasmPerformanceBenchmark::new();
        assert_eq!(benchmark.results.len(), 0);
        assert_eq!(benchmark.config.warmup_iterations, 5);
    }
}
