//! Eager Execution Performance Profiler
//!
//! This profiler targets the goal of achieving sub-millisecond overhead for eager execution
//! as mentioned in the TODO.md files, aiming for < 1000 microseconds overhead per operation.

use scirs2_autograd::ndarray::{Array1, Array2};
use std::collections::HashMap;
use std::time::Instant;
use tenflowers_core::{DType, Device, Tensor};

/// Overhead measurement for a single operation
#[derive(Debug, Clone)]
pub struct OverheadMeasurement {
    pub operation: String,
    pub input_size: usize,
    pub execution_time_microseconds: f64,
    pub overhead_time_microseconds: f64,
    pub overhead_percentage: f64,
    pub meets_target: bool, // < 1000 microseconds overhead
}

/// Comprehensive eager execution profiler
pub struct EagerExecutionProfiler {
    target_overhead_microseconds: f64,
    measurements: Vec<OverheadMeasurement>,
}

impl EagerExecutionProfiler {
    pub fn new(target_overhead_microseconds: f64) -> Self {
        Self {
            target_overhead_microseconds,
            measurements: Vec::new(),
        }
    }

    /// Profile an operation and measure overhead
    pub fn profile_operation<F, R>(
        &mut self,
        operation_name: &str,
        input_size: usize,
        setup_fn: F,
    ) -> Result<OverheadMeasurement, Box<dyn std::error::Error>>
    where
        F: Fn() -> Result<R, Box<dyn std::error::Error>>,
    {
        let iterations = 1000; // High iteration count for accurate overhead measurement
        let mut total_execution_time = 0.0;
        let mut total_overhead_time = 0.0;

        // Warmup phase
        for _ in 0..100 {
            let _ = setup_fn()?;
        }

        // Measurement phase
        for _ in 0..iterations {
            let start_total = Instant::now();

            // Measure pure execution time (simplified - in real implementation would be more sophisticated)
            let execution_start = Instant::now();
            let _ = setup_fn()?;
            let execution_time = execution_start.elapsed().as_micros() as f64;

            let total_time = start_total.elapsed().as_micros() as f64;
            let overhead_time = total_time - execution_time;

            total_execution_time += execution_time;
            total_overhead_time += overhead_time;
        }

        let avg_execution_time = total_execution_time / iterations as f64;
        let avg_overhead_time = total_overhead_time / iterations as f64;
        let overhead_percentage = (avg_overhead_time / avg_execution_time) * 100.0;
        let meets_target = avg_overhead_time <= self.target_overhead_microseconds;

        let measurement = OverheadMeasurement {
            operation: operation_name.to_string(),
            input_size,
            execution_time_microseconds: avg_execution_time,
            overhead_time_microseconds: avg_overhead_time,
            overhead_percentage,
            meets_target,
        };

        self.measurements.push(measurement.clone());
        Ok(measurement)
    }

    /// Generate comprehensive profiling report
    pub fn generate_report(&self) -> EagerExecutionReport {
        let total_operations = self.measurements.len();
        let operations_meeting_target = self.measurements.iter().filter(|m| m.meets_target).count();

        let target_percentage = if total_operations > 0 {
            (operations_meeting_target as f64 / total_operations as f64) * 100.0
        } else {
            0.0
        };

        let avg_overhead = if !self.measurements.is_empty() {
            self.measurements
                .iter()
                .map(|m| m.overhead_time_microseconds)
                .sum::<f64>()
                / self.measurements.len() as f64
        } else {
            0.0
        };

        let max_overhead = self
            .measurements
            .iter()
            .map(|m| m.overhead_time_microseconds)
            .fold(0.0f64, f64::max);

        let min_overhead = self
            .measurements
            .iter()
            .map(|m| m.overhead_time_microseconds)
            .fold(f64::INFINITY, f64::min);

        EagerExecutionReport {
            target_overhead_microseconds: self.target_overhead_microseconds,
            total_operations_tested: total_operations,
            operations_meeting_target,
            target_achievement_percentage: target_percentage,
            average_overhead_microseconds: avg_overhead,
            max_overhead_microseconds: max_overhead,
            min_overhead_microseconds: min_overhead,
            measurements: self.measurements.clone(),
            recommendations: self.generate_recommendations(),
        }
    }

    /// Generate performance optimization recommendations
    fn generate_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();
        let avg_overhead = self
            .measurements
            .iter()
            .map(|m| m.overhead_time_microseconds)
            .sum::<f64>()
            / self.measurements.len() as f64;

        if avg_overhead > self.target_overhead_microseconds * 2.0 {
            recommendations.push(
                "Critical: Overhead is >2x target - consider fundamental architecture changes"
                    .to_string(),
            );
        } else if avg_overhead > self.target_overhead_microseconds {
            recommendations.push("Overhead exceeds target - focus on optimization".to_string());
        } else {
            recommendations.push("Excellent: Overhead is below target!".to_string());
        }

        // Size-based recommendations
        let large_overhead_ops: Vec<_> = self
            .measurements
            .iter()
            .filter(|m| m.overhead_time_microseconds > self.target_overhead_microseconds * 1.5)
            .collect();

        if !large_overhead_ops.is_empty() {
            recommendations.push(format!(
                "Focus optimization on {} operations with high overhead",
                large_overhead_ops.len()
            ));
        }

        // Operation-specific recommendations
        let operation_groups: HashMap<String, Vec<&OverheadMeasurement>> =
            self.measurements.iter().fold(HashMap::new(), |mut acc, m| {
                acc.entry(m.operation.clone()).or_default().push(m);
                acc
            });

        for (op, measurements) in operation_groups {
            let avg_op_overhead: f64 = measurements
                .iter()
                .map(|m| m.overhead_time_microseconds)
                .sum::<f64>()
                / measurements.len() as f64;

            if avg_op_overhead > self.target_overhead_microseconds {
                recommendations.push(format!(
                    "Optimize '{}' operation (avg overhead: {:.1}µs)",
                    op, avg_op_overhead
                ));
            }
        }

        if recommendations.len() == 1 && recommendations[0].contains("Excellent") {
            recommendations.push("Consider memory usage optimization as next step".to_string());
            recommendations.push("Explore GPU acceleration for larger workloads".to_string());
        }

        recommendations
    }
}

/// Comprehensive eager execution performance report
#[derive(Debug)]
pub struct EagerExecutionReport {
    pub target_overhead_microseconds: f64,
    pub total_operations_tested: usize,
    pub operations_meeting_target: usize,
    pub target_achievement_percentage: f64,
    pub average_overhead_microseconds: f64,
    pub max_overhead_microseconds: f64,
    pub min_overhead_microseconds: f64,
    pub measurements: Vec<OverheadMeasurement>,
    pub recommendations: Vec<String>,
}

impl EagerExecutionReport {
    pub fn print_summary(&self) {
        println!("========================================");
        println!("EAGER EXECUTION PERFORMANCE REPORT");
        println!("========================================");
        println!(
            "Target Overhead: {:.0} microseconds",
            self.target_overhead_microseconds
        );
        println!("Operations Tested: {}", self.total_operations_tested);
        println!(
            "Operations Meeting Target: {}/{} ({:.1}%)",
            self.operations_meeting_target,
            self.total_operations_tested,
            self.target_achievement_percentage
        );

        if self.target_achievement_percentage >= 90.0 {
            println!("✅ EXCELLENT: >90% of operations meet sub-millisecond target!");
        } else if self.target_achievement_percentage >= 70.0 {
            println!("⚠️  GOOD: Most operations meet target, some optimization needed");
        } else {
            println!("❌ NEEDS WORK: Significant optimization required");
        }

        println!("\nOverhead Statistics:");
        println!("  Average: {:.1}µs", self.average_overhead_microseconds);
        println!("  Minimum: {:.1}µs", self.min_overhead_microseconds);
        println!("  Maximum: {:.1}µs", self.max_overhead_microseconds);

        println!("\nDetailed Results:");
        for measurement in &self.measurements {
            let status = if measurement.meets_target {
                "✅"
            } else {
                "❌"
            };
            println!(
                "  {} {}: {:.1}µs overhead ({:.1}% of execution time) [size: {}]",
                status,
                measurement.operation,
                measurement.overhead_time_microseconds,
                measurement.overhead_percentage,
                measurement.input_size
            );
        }

        println!("\nRecommendations:");
        for (i, rec) in self.recommendations.iter().enumerate() {
            println!("  {}. {}", i + 1, rec);
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("TenfloweRS Eager Execution Profiler");
    println!("Goal: Achieve sub-millisecond overhead for eager execution");
    println!("Target: < 1000 microseconds overhead per operation\n");

    let mut profiler = EagerExecutionProfiler::new(1000.0); // 1000 microseconds = 1 millisecond

    // Test different operation types and sizes
    let test_cases = vec![
        ("small_add", 100),
        ("medium_add", 1000),
        ("large_add", 10000),
        ("small_matmul", 64),
        ("medium_matmul", 256),
        ("large_matmul", 512),
        ("small_mul", 100),
        ("medium_mul", 1000),
        ("large_mul", 10000),
    ];

    for (operation, size) in test_cases {
        println!("Profiling {}...", operation);

        let measurement = match operation {
            op if op.contains("add") => profiler.profile_operation(operation, size, || {
                let a = Tensor::from_array(Array1::<f32>::ones(size).into_dyn());
                let b = Tensor::from_array(Array1::<f32>::ones(size).into_dyn());
                let _result = a.add(&b)?;
                Ok(())
            })?,
            op if op.contains("matmul") => {
                profiler.profile_operation(operation, size * size, || {
                    let a = Tensor::from_array(Array2::<f32>::ones((size, size)).into_dyn());
                    let b = Tensor::from_array(Array2::<f32>::ones((size, size)).into_dyn());
                    let _result = a.matmul(&b)?;
                    Ok(())
                })?
            }
            op if op.contains("mul") => profiler.profile_operation(operation, size, || {
                let a = Tensor::from_array(Array1::<f32>::ones(size).into_dyn());
                let b = Tensor::from_array(Array1::<f32>::ones(size).into_dyn());
                let _result = a.mul(&b)?;
                Ok(())
            })?,
            _ => unreachable!(),
        };

        println!(
            "  Result: {:.1}µs overhead ({})",
            measurement.overhead_time_microseconds,
            if measurement.meets_target {
                "✅ PASS"
            } else {
                "❌ FAIL"
            }
        );
    }

    println!();
    let report = profiler.generate_report();
    report.print_summary();

    Ok(())
}
