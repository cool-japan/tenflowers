/// Performance Regression Gates
///
/// This module provides infrastructure for performance regression testing using
/// criterion-based thresholds. It ensures that performance-critical operations
/// maintain their speed characteristics across code changes.
///
/// # Architecture
///
/// The performance gate system consists of:
/// - Baseline measurements for critical operations
/// - Configurable regression thresholds
/// - Automatic validation against baselines
/// - CI-friendly pass/fail reporting
///
/// # Usage
///
/// ```rust
/// use tenflowers_core::performance_gates::{PerformanceGate, OperationBaseline};
/// use tenflowers_core::{Tensor, ops::matmul};
///
/// // Define a baseline
/// let baseline = OperationBaseline::new(
///     "matmul_64x64",
///     100_000, // 100 microseconds baseline
///     0.10,    // Allow 10% regression
/// );
///
/// // Create test data
/// let size = 64;
/// let a = Tensor::<f32>::zeros(&[size, size]);
/// let b = Tensor::<f32>::zeros(&[size, size]);
///
/// // Validate performance
/// let gate = PerformanceGate::new(baseline);
/// let passed = gate.validate(|| {
///     matmul(&a, &b).unwrap();
/// });
///
/// assert!(passed, "Performance regression detected!");
/// ```
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

lazy_static::lazy_static! {
    /// Global registry of performance baselines
    static ref PERFORMANCE_BASELINES: Arc<Mutex<HashMap<String, OperationBaseline>>> = {
        Arc::new(Mutex::new(initialize_baselines()))
    };
}

/// Performance baseline for a specific operation
#[derive(Debug, Clone)]
pub struct OperationBaseline {
    /// Name of the operation
    pub name: String,
    /// Baseline time in nanoseconds
    pub baseline_ns: u64,
    /// Maximum allowed regression (0.10 = 10%)
    pub max_regression: f64,
    /// Minimum sample size for measurements
    pub min_samples: usize,
    /// Warmup iterations before measurement
    pub warmup_iters: usize,
}

impl OperationBaseline {
    /// Create a new operation baseline
    pub fn new(name: &str, baseline_ns: u64, max_regression: f64) -> Self {
        Self {
            name: name.to_string(),
            baseline_ns,
            max_regression,
            min_samples: 10,
            warmup_iters: 3,
        }
    }

    /// Create a baseline with custom sampling configuration
    pub fn with_sampling(
        name: &str,
        baseline_ns: u64,
        max_regression: f64,
        min_samples: usize,
        warmup_iters: usize,
    ) -> Self {
        Self {
            name: name.to_string(),
            baseline_ns,
            max_regression,
            min_samples,
            warmup_iters,
        }
    }

    /// Check if a measured time passes the regression threshold
    pub fn check_regression(&self, measured_ns: u64) -> bool {
        let threshold_ns = (self.baseline_ns as f64 * (1.0 + self.max_regression)) as u64;
        measured_ns <= threshold_ns
    }

    /// Calculate regression percentage
    pub fn regression_percentage(&self, measured_ns: u64) -> f64 {
        ((measured_ns as f64 - self.baseline_ns as f64) / self.baseline_ns as f64) * 100.0
    }
}

/// Performance gate validator
pub struct PerformanceGate {
    baseline: OperationBaseline,
}

impl PerformanceGate {
    /// Create a new performance gate
    pub fn new(baseline: OperationBaseline) -> Self {
        Self { baseline }
    }

    /// Validate that operation meets performance baseline
    ///
    /// Returns true if performance is within acceptable regression threshold
    pub fn validate<F>(&self, mut op: F) -> bool
    where
        F: FnMut(),
    {
        // Warmup iterations
        for _ in 0..self.baseline.warmup_iters {
            op();
        }

        // Measurement iterations
        let mut times = Vec::with_capacity(self.baseline.min_samples);
        for _ in 0..self.baseline.min_samples {
            let start = Instant::now();
            op();
            let elapsed = start.elapsed();
            times.push(elapsed.as_nanos() as u64);
        }

        // Calculate median time (more robust than mean for performance)
        times.sort_unstable();
        let median_ns = times[times.len() / 2];

        self.baseline.check_regression(median_ns)
    }

    /// Validate and return detailed measurement
    pub fn validate_detailed<F>(&self, mut op: F) -> PerformanceMeasurement
    where
        F: FnMut(),
    {
        // Warmup iterations
        for _ in 0..self.baseline.warmup_iters {
            op();
        }

        // Measurement iterations
        let mut times = Vec::with_capacity(self.baseline.min_samples);
        for _ in 0..self.baseline.min_samples {
            let start = Instant::now();
            op();
            let elapsed = start.elapsed();
            times.push(elapsed.as_nanos() as u64);
        }

        // Calculate statistics
        times.sort_unstable();
        let median_ns = times[times.len() / 2];
        let min_ns = *times.first().unwrap();
        let max_ns = *times.last().unwrap();
        let mean_ns = times.iter().sum::<u64>() / times.len() as u64;

        let passed = self.baseline.check_regression(median_ns);
        let regression_pct = self.baseline.regression_percentage(median_ns);

        PerformanceMeasurement {
            operation: self.baseline.name.clone(),
            baseline_ns: self.baseline.baseline_ns,
            measured_ns: median_ns,
            min_ns,
            max_ns,
            mean_ns,
            regression_pct,
            passed,
            samples: times.len(),
        }
    }
}

/// Detailed performance measurement result
#[derive(Debug, Clone)]
pub struct PerformanceMeasurement {
    pub operation: String,
    pub baseline_ns: u64,
    pub measured_ns: u64,
    pub min_ns: u64,
    pub max_ns: u64,
    pub mean_ns: u64,
    pub regression_pct: f64,
    pub passed: bool,
    pub samples: usize,
}

impl PerformanceMeasurement {
    /// Format as human-readable report
    pub fn report(&self) -> String {
        let status = if self.passed { "✓ PASS" } else { "✗ FAIL" };
        let regression_sign = if self.regression_pct >= 0.0 { "+" } else { "" };

        format!(
            "{} | {} | baseline: {:>8}ns | measured: {:>8}ns | regression: {}{:>6.2}% | min: {:>8}ns | max: {:>8}ns | samples: {}",
            status,
            self.operation,
            self.baseline_ns,
            self.measured_ns,
            regression_sign,
            self.regression_pct,
            self.min_ns,
            self.max_ns,
            self.samples
        )
    }

    /// Get duration from nanoseconds
    pub fn baseline_duration(&self) -> Duration {
        Duration::from_nanos(self.baseline_ns)
    }

    /// Get measured duration
    pub fn measured_duration(&self) -> Duration {
        Duration::from_nanos(self.measured_ns)
    }
}

/// Suite of performance gates for comprehensive validation
pub struct PerformanceGateSuite {
    gates: Vec<(String, PerformanceGate)>,
}

impl PerformanceGateSuite {
    /// Create a new empty suite
    pub fn new() -> Self {
        Self { gates: Vec::new() }
    }

    /// Add a gate to the suite
    pub fn add_gate(&mut self, name: String, gate: PerformanceGate) -> &mut Self {
        self.gates.push((name, gate));
        self
    }

    /// Run all gates and collect results
    pub fn run_all<F>(&self, op_factory: F) -> Vec<PerformanceMeasurement>
    where
        F: Fn(&str) -> Box<dyn FnMut()>,
    {
        let mut results = Vec::new();
        for (name, gate) in &self.gates {
            let mut op = op_factory(name);
            let measurement = gate.validate_detailed(&mut *op);
            results.push(measurement);
        }
        results
    }

    /// Check if all gates pass
    pub fn all_passed(&self, results: &[PerformanceMeasurement]) -> bool {
        results.iter().all(|r| r.passed)
    }

    /// Print comprehensive report
    pub fn print_report(&self, results: &[PerformanceMeasurement]) {
        println!(
            "\n╔════════════════════════════════════════════════════════════════════════════╗"
        );
        println!("║                    PERFORMANCE REGRESSION GATE REPORT                      ║");
        println!(
            "╚════════════════════════════════════════════════════════════════════════════╝\n"
        );

        for result in results {
            println!("{}", result.report());
        }

        let total = results.len();
        let passed = results.iter().filter(|r| r.passed).count();
        let failed = total - passed;

        println!("\n{}", "─".repeat(80));
        println!(
            "Summary: {} total | {} passed | {} failed",
            total, passed, failed
        );

        if failed > 0 {
            println!("\n⚠ WARNING: Performance regressions detected!");
        } else {
            println!("\n✓ All performance gates passed!");
        }
    }
}

impl Default for PerformanceGateSuite {
    fn default() -> Self {
        Self::new()
    }
}

/// Initialize default performance baselines for critical operations
fn initialize_baselines() -> HashMap<String, OperationBaseline> {
    let mut baselines = HashMap::new();

    // Matrix multiplication baselines (in nanoseconds)
    // These are conservative estimates - adjust based on your hardware
    baselines.insert(
        "matmul_64x64_f32".to_string(),
        OperationBaseline::new("matmul_64x64_f32", 50_000, 0.15),
    );
    baselines.insert(
        "matmul_128x128_f32".to_string(),
        OperationBaseline::new("matmul_128x128_f32", 400_000, 0.15),
    );
    baselines.insert(
        "matmul_256x256_f32".to_string(),
        OperationBaseline::new("matmul_256x256_f32", 3_000_000, 0.15),
    );

    // Binary operations baselines
    baselines.insert(
        "add_10k_f32".to_string(),
        OperationBaseline::new("add_10k_f32", 5_000, 0.20),
    );
    baselines.insert(
        "mul_10k_f32".to_string(),
        OperationBaseline::new("mul_10k_f32", 5_000, 0.20),
    );

    // Reduction operations baselines
    baselines.insert(
        "sum_100k_f32".to_string(),
        OperationBaseline::new("sum_100k_f32", 20_000, 0.20),
    );
    baselines.insert(
        "mean_100k_f32".to_string(),
        OperationBaseline::new("mean_100k_f32", 25_000, 0.20),
    );

    // Convolution baselines (conservative for CPU)
    baselines.insert(
        "conv2d_3x3_32ch".to_string(),
        OperationBaseline::new("conv2d_3x3_32ch", 1_000_000, 0.15),
    );

    baselines
}

/// Register a custom baseline
pub fn register_baseline(baseline: OperationBaseline) {
    if let Ok(mut baselines) = PERFORMANCE_BASELINES.lock() {
        baselines.insert(baseline.name.clone(), baseline);
    }
}

/// Get a baseline by name
pub fn get_baseline(name: &str) -> Option<OperationBaseline> {
    PERFORMANCE_BASELINES
        .lock()
        .ok()
        .and_then(|baselines| baselines.get(name).cloned())
}

/// List all registered baselines
pub fn list_baselines() -> Vec<String> {
    PERFORMANCE_BASELINES
        .lock()
        .ok()
        .map(|baselines| baselines.keys().cloned().collect())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_baseline_creation() {
        let baseline = OperationBaseline::new("test_op", 1000, 0.10);
        assert_eq!(baseline.name, "test_op");
        assert_eq!(baseline.baseline_ns, 1000);
        assert_eq!(baseline.max_regression, 0.10);
    }

    #[test]
    fn test_regression_check() {
        let baseline = OperationBaseline::new("test_op", 1000, 0.10);

        // Within threshold
        assert!(baseline.check_regression(1000));
        assert!(baseline.check_regression(1050));
        assert!(baseline.check_regression(1100));

        // Exceeds threshold
        assert!(!baseline.check_regression(1150));
        assert!(!baseline.check_regression(2000));
    }

    #[test]
    fn test_regression_percentage() {
        let baseline = OperationBaseline::new("test_op", 1000, 0.10);

        assert_eq!(baseline.regression_percentage(1000), 0.0);
        assert_eq!(baseline.regression_percentage(1100), 10.0);
        assert_eq!(baseline.regression_percentage(1200), 20.0);
        assert_eq!(baseline.regression_percentage(900), -10.0);
    }

    #[test]
    fn test_performance_gate_validation() {
        // Use more realistic baseline accounting for closure call and measurement overhead
        let baseline = OperationBaseline::new("fast_op", 10_000, 0.50);
        let gate = PerformanceGate::new(baseline);

        // Fast operation should pass
        let passed = gate.validate(|| {
            // Simulate fast operation (essentially instant)
            let _ = 1 + 1;
        });

        assert!(passed, "Fast operation should pass performance gate");
    }

    #[test]
    fn test_performance_measurement_report() {
        let measurement = PerformanceMeasurement {
            operation: "test_op".to_string(),
            baseline_ns: 1000,
            measured_ns: 1050,
            min_ns: 1000,
            max_ns: 1100,
            mean_ns: 1050,
            regression_pct: 5.0,
            passed: true,
            samples: 10,
        };

        let report = measurement.report();
        assert!(report.contains("✓ PASS"));
        assert!(report.contains("test_op"));
        assert!(report.contains("1000"));
        assert!(report.contains("1050"));
    }

    #[test]
    fn test_baseline_registry() {
        let baseline = OperationBaseline::new("custom_op", 5000, 0.15);
        register_baseline(baseline.clone());

        let retrieved = get_baseline("custom_op");
        assert!(retrieved.is_some());
        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.name, "custom_op");
        assert_eq!(retrieved.baseline_ns, 5000);
    }

    #[test]
    fn test_gate_suite() {
        let mut suite = PerformanceGateSuite::new();

        // Use more realistic baselines accounting for closure call and measurement overhead
        let baseline1 = OperationBaseline::new("op1", 10_000, 0.50);
        let baseline2 = OperationBaseline::new("op2", 10_000, 0.50);

        suite.add_gate("op1".to_string(), PerformanceGate::new(baseline1));
        suite.add_gate("op2".to_string(), PerformanceGate::new(baseline2));

        let results = suite.run_all(|_name| {
            Box::new(|| {
                let _ = 1 + 1;
            })
        });

        assert_eq!(results.len(), 2);
        assert!(suite.all_passed(&results));
    }
}
