/// Performance Gate Validation Benchmark
///
/// This benchmark validates that critical operations meet performance baselines
/// and can be used in CI to catch performance regressions.
use tenflowers_core::performance_gates::{
    OperationBaseline, PerformanceGate, PerformanceGateSuite,
};
use tenflowers_core::{
    ops::{add, matmul, mean, mul, sum},
    Tensor,
};

fn main() {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║         TenfloweRS Performance Regression Validation        ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Create suite of performance gates
    let mut suite = PerformanceGateSuite::new();

    // Matrix multiplication gates
    add_matmul_gates(&mut suite);

    // Binary operation gates
    add_binary_op_gates(&mut suite);

    // Reduction operation gates
    add_reduction_gates(&mut suite);

    // Run all gates
    let results = suite.run_all(|name| create_operation_closure(name));

    // Print comprehensive report
    suite.print_report(&results);

    // Exit with error code if any gate failed (for CI integration)
    if !suite.all_passed(&results) {
        std::process::exit(1);
    }
}

/// Add matrix multiplication performance gates
fn add_matmul_gates(suite: &mut PerformanceGateSuite) {
    // Small matmul (64x64)
    let baseline_64 = OperationBaseline::new("matmul_64x64_f32", 100_000, 0.20);
    suite.add_gate(
        "matmul_64x64_f32".to_string(),
        PerformanceGate::new(baseline_64),
    );

    // Medium matmul (128x128)
    let baseline_128 = OperationBaseline::new("matmul_128x128_f32", 800_000, 0.20);
    suite.add_gate(
        "matmul_128x128_f32".to_string(),
        PerformanceGate::new(baseline_128),
    );

    // Large matmul (256x256)
    let baseline_256 = OperationBaseline::new("matmul_256x256_f32", 6_000_000, 0.20);
    suite.add_gate(
        "matmul_256x256_f32".to_string(),
        PerformanceGate::new(baseline_256),
    );
}

/// Add binary operation performance gates
fn add_binary_op_gates(suite: &mut PerformanceGateSuite) {
    // Addition (10k elements)
    let baseline_add = OperationBaseline::new("add_10k_f32", 10_000, 0.25);
    suite.add_gate(
        "add_10k_f32".to_string(),
        PerformanceGate::new(baseline_add),
    );

    // Multiplication (10k elements)
    let baseline_mul = OperationBaseline::new("mul_10k_f32", 10_000, 0.25);
    suite.add_gate(
        "mul_10k_f32".to_string(),
        PerformanceGate::new(baseline_mul),
    );

    // Large addition (100k elements)
    let baseline_add_100k = OperationBaseline::new("add_100k_f32", 50_000, 0.25);
    suite.add_gate(
        "add_100k_f32".to_string(),
        PerformanceGate::new(baseline_add_100k),
    );
}

/// Add reduction operation performance gates
fn add_reduction_gates(suite: &mut PerformanceGateSuite) {
    // Sum (100k elements)
    let baseline_sum = OperationBaseline::new("sum_100k_f32", 30_000, 0.25);
    suite.add_gate(
        "sum_100k_f32".to_string(),
        PerformanceGate::new(baseline_sum),
    );

    // Mean (100k elements)
    let baseline_mean = OperationBaseline::new("mean_100k_f32", 35_000, 0.25);
    suite.add_gate(
        "mean_100k_f32".to_string(),
        PerformanceGate::new(baseline_mean),
    );
}

/// Create operation closure for a given test name
fn create_operation_closure(name: &str) -> Box<dyn FnMut()> {
    match name {
        // Matrix multiplication operations
        "matmul_64x64_f32" => {
            let size = 64;
            let a = Tensor::<f32>::ones(&[size, size]);
            let b = Tensor::<f32>::ones(&[size, size]);
            Box::new(move || {
                let _ = matmul(&a, &b).unwrap();
            })
        }
        "matmul_128x128_f32" => {
            let size = 128;
            let a = Tensor::<f32>::ones(&[size, size]);
            let b = Tensor::<f32>::ones(&[size, size]);
            Box::new(move || {
                let _ = matmul(&a, &b).unwrap();
            })
        }
        "matmul_256x256_f32" => {
            let size = 256;
            let a = Tensor::<f32>::ones(&[size, size]);
            let b = Tensor::<f32>::ones(&[size, size]);
            Box::new(move || {
                let _ = matmul(&a, &b).unwrap();
            })
        }

        // Binary operations
        "add_10k_f32" => {
            let size = 10_000;
            let a = Tensor::<f32>::ones(&[size]);
            let b = Tensor::<f32>::ones(&[size]);
            Box::new(move || {
                let _ = add(&a, &b).unwrap();
            })
        }
        "mul_10k_f32" => {
            let size = 10_000;
            let a = Tensor::<f32>::ones(&[size]);
            let b = Tensor::<f32>::ones(&[size]);
            Box::new(move || {
                let _ = mul(&a, &b).unwrap();
            })
        }
        "add_100k_f32" => {
            let size = 100_000;
            let a = Tensor::<f32>::ones(&[size]);
            let b = Tensor::<f32>::ones(&[size]);
            Box::new(move || {
                let _ = add(&a, &b).unwrap();
            })
        }

        // Reduction operations
        "sum_100k_f32" => {
            let size = 100_000;
            let a = Tensor::<f32>::ones(&[size]);
            Box::new(move || {
                let _ = sum(&a, None, false).unwrap();
            })
        }
        "mean_100k_f32" => {
            let size = 100_000;
            let a = Tensor::<f32>::ones(&[size]);
            Box::new(move || {
                let _ = mean(&a, None, false).unwrap();
            })
        }

        _ => {
            panic!("Unknown operation: {}", name);
        }
    }
}
