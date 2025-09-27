use std::time::Duration;
use tenflowers_core::{
    ops::benchmark::{benchmark_binary_op, benchmark_unary_op, BenchmarkConfig, BenchmarkSuite},
    DType, Device, Tensor,
};

#[test]
fn test_benchmark_infrastructure_integration() {
    // Test basic benchmark configuration
    let config = BenchmarkConfig {
        warmup_iterations: 2,
        measurement_iterations: 3,
        measure_memory: false, // Disable for faster test
        calculate_flops: true,
        min_execution_time: Duration::from_micros(1),
        max_execution_time: Duration::from_secs(5),
    };

    // Test benchmark suite creation
    let suite = BenchmarkSuite::new(config);

    // Test devices (just CPU for integration test)
    let devices = vec![Device::Cpu];

    // Test simple binary operation benchmark
    let results =
        benchmark_binary_op::<f32>("Add", &[10, 10], &[10, 10], &devices, &[DType::Float32]);
    assert!(results.is_ok(), "Binary op benchmark should succeed");

    let results = results.unwrap();
    assert!(!results.is_empty(), "Should have benchmark results");

    for result in &results {
        assert_eq!(result.operation, "Add");
        assert_eq!(result.device, Device::Cpu);
        assert_eq!(result.dtype, DType::Float32);
        assert_eq!(result.input_shapes.len(), 2);
        assert!(
            result.duration.as_nanos() > 0,
            "Duration should be positive"
        );

        // Check that throughput was calculated
        assert!(
            result.throughput.is_some(),
            "Throughput should be calculated"
        );
        assert!(
            result.throughput.unwrap() > 0.0,
            "Throughput should be positive"
        );
    }

    // Test unary operation benchmark
    let unary_results = benchmark_unary_op::<f32>("ReLU", &[20, 20], &devices, &[DType::Float32]);
    assert!(unary_results.is_ok(), "Unary op benchmark should succeed");

    let unary_results = unary_results.unwrap();
    assert!(
        !unary_results.is_empty(),
        "Should have unary benchmark results"
    );

    for result in &unary_results {
        assert_eq!(result.operation, "ReLU");
        assert_eq!(result.input_shapes.len(), 1);
    }

    // Test manual tensor benchmarking
    let tensor_a: Tensor<f32> = Tensor::ones(&[15, 15]);
    let tensor_b: Tensor<f32> = Tensor::ones(&[15, 15]);

    let inputs = vec![&tensor_a, &tensor_b];
    let attrs = std::collections::HashMap::new();

    let manual_result = suite.benchmark_operation("Add", &inputs, &attrs);
    assert!(manual_result.is_ok(), "Manual benchmark should succeed");

    let manual_result = manual_result.unwrap();
    assert_eq!(manual_result.operation, "Add");
    assert!(manual_result.flops.is_some(), "FLOPS should be calculated");
    assert!(
        manual_result.throughput.is_some(),
        "Throughput should be calculated"
    );

    // Test report generation
    let report = suite.generate_report();
    assert!(!report.is_empty(), "Report should not be empty");
    assert!(
        report.contains("Add"),
        "Report should mention Add operation"
    );
}

#[test]
fn test_benchmark_suite_functionality() {
    let suite = BenchmarkSuite::new_default();

    // Test that we can create benchmark results and they contain FLOPS
    let tensor_a: Tensor<f32> = Tensor::ones(&[50, 50]);
    let tensor_b: Tensor<f32> = Tensor::ones(&[50, 50]);

    let inputs = vec![&tensor_a, &tensor_b];
    let attrs = std::collections::HashMap::new();

    // Test Add operation
    let add_result = suite.benchmark_operation("Add", &inputs, &attrs);
    assert!(add_result.is_ok(), "Add benchmark should succeed");

    let add_result = add_result.unwrap();
    assert_eq!(add_result.operation, "Add");
    assert!(
        add_result.flops.is_some(),
        "Add should have FLOPS calculated"
    );
    assert_eq!(
        add_result.flops.unwrap(),
        2500.0,
        "Add should have 1 FLOP per element"
    );

    // Test MatMul operation
    let tensor_c: Tensor<f32> = Tensor::ones(&[32, 64]);
    let tensor_d: Tensor<f32> = Tensor::ones(&[64, 128]);
    let matmul_inputs = vec![&tensor_c, &tensor_d];

    let matmul_result = suite.benchmark_operation("MatMul", &matmul_inputs, &attrs);
    assert!(matmul_result.is_ok(), "MatMul benchmark should succeed");

    let matmul_result = matmul_result.unwrap();
    assert_eq!(matmul_result.operation, "MatMul");
    assert!(
        matmul_result.flops.is_some(),
        "MatMul should have FLOPS calculated"
    );

    // FLOPS for MatMul should be 2 * M * N * K = 2 * 32 * 128 * 64
    let expected_flops = 2.0 * 32.0 * 128.0 * 64.0;
    assert_eq!(
        matmul_result.flops.unwrap(),
        expected_flops,
        "MatMul FLOPS should be correct"
    );
}

#[test]
fn test_benchmark_config_validation() {
    let config = BenchmarkConfig::default();
    assert!(config.warmup_iterations > 0);
    assert!(config.measurement_iterations > 0);
    assert!(config.min_execution_time.as_nanos() > 0);
    assert!(config.max_execution_time > config.min_execution_time);
}

#[test]
fn test_benchmark_error_handling() {
    let suite = BenchmarkSuite::new_default();

    // Test with invalid operation name
    let tensor: Tensor<f32> = Tensor::ones(&[10, 10]);
    let inputs = vec![&tensor];
    let attrs = std::collections::HashMap::new();

    let result = suite.benchmark_operation("NonExistentOp", &inputs, &attrs);
    assert!(result.is_err(), "Should fail for non-existent operation");
}
