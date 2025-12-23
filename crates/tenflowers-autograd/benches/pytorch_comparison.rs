use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray::{Array1, Array2, Array3, Array4};
use serde_json::{json, Value};
use std::process::Command;
use std::time::{Duration, Instant};
use tenflowers_autograd::{GradientAccumulator, GradientTape};
use tenflowers_core::{DType, Device, Tensor};

/// Comprehensive benchmarking suite for comparing TenfloweRS autograd performance
/// with PyTorch and TensorFlow. This provides standardized tests that can be
/// replicated across different frameworks.
///
/// To enable external comparisons, run the companion Python scripts:
/// - scripts/pytorch_benchmark.py
/// - scripts/tensorflow_benchmark.py

/// Standard benchmark configurations for cross-framework comparison
#[derive(Clone, Debug)]
pub struct BenchmarkConfig {
    pub name: String,
    pub input_shapes: Vec<Vec<usize>>,
    pub operation_chain_length: usize,
    pub batch_sizes: Vec<usize>,
    pub warmup_iterations: usize,
    pub measurement_iterations: usize,
}

impl BenchmarkConfig {
    pub fn new(name: String) -> Self {
        Self {
            name,
            input_shapes: vec![vec![100], vec![1000], vec![10000]],
            operation_chain_length: 5,
            batch_sizes: vec![1, 8, 32, 128],
            warmup_iterations: 10,
            measurement_iterations: 100,
        }
    }

    pub fn with_shapes(mut self, shapes: Vec<Vec<usize>>) -> Self {
        self.input_shapes = shapes;
        self
    }

    pub fn with_batch_sizes(mut self, batch_sizes: Vec<usize>) -> Self {
        self.batch_sizes = batch_sizes;
        self
    }

    pub fn with_chain_length(mut self, length: usize) -> Self {
        self.operation_chain_length = length;
        self
    }
}

/// Benchmark result for cross-framework comparison
#[derive(Clone, Debug)]
pub struct BenchmarkResult {
    pub framework: String,
    pub operation: String,
    pub input_size: usize,
    pub batch_size: usize,
    pub forward_time_ms: f64,
    pub backward_time_ms: f64,
    pub total_time_ms: f64,
    pub memory_usage_mb: Option<f64>,
    pub throughput_ops_per_sec: f64,
}

impl BenchmarkResult {
    pub fn new(
        framework: String,
        operation: String,
        input_size: usize,
        batch_size: usize,
        total_time_ms: f64,
    ) -> Self {
        Self {
            framework,
            operation,
            input_size,
            batch_size,
            forward_time_ms: 0.0,
            backward_time_ms: 0.0,
            total_time_ms,
            memory_usage_mb: None,
            throughput_ops_per_sec: 1000.0 / total_time_ms,
        }
    }

    pub fn with_forward_backward_split(mut self, forward_ms: f64, backward_ms: f64) -> Self {
        self.forward_time_ms = forward_ms;
        self.backward_time_ms = backward_ms;
        self
    }

    pub fn with_memory_usage(mut self, memory_mb: f64) -> Self {
        self.memory_usage_mb = Some(memory_mb);
        self
    }

    pub fn to_json(&self) -> Value {
        json!({
            "framework": self.framework,
            "operation": self.operation,
            "input_size": self.input_size,
            "batch_size": self.batch_size,
            "forward_time_ms": self.forward_time_ms,
            "backward_time_ms": self.backward_time_ms,
            "total_time_ms": self.total_time_ms,
            "memory_usage_mb": self.memory_usage_mb,
            "throughput_ops_per_sec": self.throughput_ops_per_sec
        })
    }
}

/// Benchmark basic operations against PyTorch equivalent
fn bench_basic_operations_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("basic_operations_comparison");

    let config = BenchmarkConfig::new("basic_ops".to_string()).with_shapes(vec![
        vec![1000],
        vec![10000],
        vec![100000],
    ]);

    for shape in &config.input_shapes {
        let size = shape[0];
        group.throughput(Throughput::Elements(size as u64));

        // Element-wise addition
        group.bench_with_input(
            BenchmarkId::new("add_elementwise", size),
            &size,
            |b, &size| {
                let tape = GradientTape::new();
                let x_data = Array1::linspace(0.0f32, 1.0, size).into_dyn();
                let y_data = Array1::linspace(1.0f32, 2.0, size).into_dyn();
                let x = tape.watch(Tensor::from_array(x_data));
                let y = tape.watch(Tensor::from_array(y_data));

                b.iter_custom(|iters| {
                    let start = Instant::now();
                    for _ in 0..iters {
                        let z = x.add(&y).unwrap();
                        let _grads = tape.gradient(&z, &[&x, &y]).unwrap();
                        black_box(z);
                    }
                    start.elapsed()
                });
            },
        );

        // Element-wise multiplication
        group.bench_with_input(
            BenchmarkId::new("mul_elementwise", size),
            &size,
            |b, &size| {
                let tape = GradientTape::new();
                let x_data = Array1::linspace(0.1f32, 1.0, size).into_dyn();
                let y_data = Array1::linspace(1.0f32, 2.0, size).into_dyn();
                let x = tape.watch(Tensor::from_array(x_data));
                let y = tape.watch(Tensor::from_array(y_data));

                b.iter_custom(|iters| {
                    let start = Instant::now();
                    for _ in 0..iters {
                        let z = x.mul(&y).unwrap();
                        let _grads = tape.gradient(&z, &[&x, &y]).unwrap();
                        black_box(z);
                    }
                    start.elapsed()
                });
            },
        );

        // Power operation
        group.bench_with_input(BenchmarkId::new("pow", size), &size, |b, &size| {
            let tape = GradientTape::new();
            let x_data = Array1::linspace(0.1f32, 2.0, size).into_dyn();
            let x = tape.watch(Tensor::from_array(x_data));
            let exponent = tape.watch(Tensor::from_scalar(2.0f32));

            b.iter_custom(|iters| {
                let start = Instant::now();
                for _ in 0..iters {
                    let z = x.pow(&exponent).unwrap();
                    let _grads = tape.gradient(&z, &[&x]).unwrap();
                    black_box(z);
                }
                start.elapsed()
            });
        });
    }

    group.finish();
}

/// Benchmark matrix operations against PyTorch
fn bench_matrix_operations_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_operations_comparison");

    let matrix_sizes = vec![64, 128, 256, 512];

    for size in matrix_sizes {
        group.throughput(Throughput::Elements((size * size) as u64));

        // Matrix multiplication
        group.bench_with_input(BenchmarkId::new("matmul", size), &size, |b, &size| {
            let tape = GradientTape::new();
            let x_data = Array2::zeros((size, size)).into_dyn();
            let y_data = Array2::eye(size).into_dyn();
            let x = tape.watch(Tensor::from_array(x_data));
            let y = tape.watch(Tensor::from_array(y_data));

            b.iter_custom(|iters| {
                let start = Instant::now();
                for _ in 0..iters {
                    let z = x.matmul(&y).unwrap();
                    let _grads = tape.gradient(&z, &[&x, &y]).unwrap();
                    black_box(z);
                }
                start.elapsed()
            });
        });

        // Matrix transpose
        group.bench_with_input(BenchmarkId::new("transpose", size), &size, |b, &size| {
            let tape = GradientTape::new();
            let x_data = Array2::zeros((size, size)).into_dyn();
            let x = tape.watch(Tensor::from_array(x_data));

            b.iter_custom(|iters| {
                let start = Instant::now();
                for _ in 0..iters {
                    let z = x.transpose().unwrap();
                    let _grads = tape.gradient(&z, &[&x]).unwrap();
                    black_box(z);
                }
                start.elapsed()
            });
        });

        // Matrix determinant (if available)
        if size <= 256 {
            // Limit size for determinant computation
            group.bench_with_input(BenchmarkId::new("determinant", size), &size, |b, &size| {
                let tape = GradientTape::new();
                let x_data = Array2::eye(size).into_dyn();
                let x = tape.watch(Tensor::from_array(x_data));

                b.iter_custom(|iters| {
                    let start = Instant::now();
                    for _ in 0..iters {
                        // Note: This would require det operation to be implemented
                        // For now, use a placeholder operation
                        let z = x.sum(None, false).unwrap();
                        let _grads = tape.gradient(&z, &[&x]).unwrap();
                        black_box(z);
                    }
                    start.elapsed()
                });
            });
        }
    }

    group.finish();
}

/// Benchmark neural network operations
fn bench_neural_network_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("neural_network_comparison");

    let configs = vec![
        (32, 784, 128, 10),   // MNIST-like
        (64, 2048, 512, 100), // Larger network
        (128, 1024, 256, 50), // Medium network
    ];

    for (batch_size, input_size, hidden_size, output_size) in configs {
        let config_name = format!(
            "mlp_{}x{}x{}x{}",
            batch_size, input_size, hidden_size, output_size
        );

        group.bench_with_input(
            BenchmarkId::new("mlp_forward_backward", &config_name),
            &(batch_size, input_size, hidden_size, output_size),
            |b, &(batch_size, input_size, hidden_size, output_size)| {
                let tape = GradientTape::new();

                // Create network parameters
                let x_data = Array2::<f32>::zeros((batch_size, input_size)).into_dyn();
                let w1_data = Array2::<f32>::zeros((input_size, hidden_size)).into_dyn();
                let b1_data = Array1::<f32>::zeros(hidden_size).into_dyn();
                let w2_data = Array2::<f32>::zeros((hidden_size, output_size)).into_dyn();
                let b2_data = Array1::<f32>::zeros(output_size).into_dyn();

                let x = tape.watch(Tensor::from_array(x_data));
                let w1 = tape.watch(Tensor::from_array(w1_data));
                let b1 = tape.watch(Tensor::from_array(b1_data));
                let w2 = tape.watch(Tensor::from_array(w2_data));
                let b2 = tape.watch(Tensor::from_array(b2_data));

                b.iter_custom(|iters| {
                    let start = Instant::now();
                    for _ in 0..iters {
                        // Forward pass
                        let h1 = x.matmul(&w1).unwrap().add(&b1).unwrap();
                        let h1_relu = h1.relu().unwrap();
                        let h2 = h1_relu.matmul(&w2).unwrap().add(&b2).unwrap();
                        let output = h2.softmax(Some(1)).unwrap();
                        let loss = output.sum(None, false).unwrap();

                        // Backward pass
                        let _grads = tape.gradient(&loss, &[&w1, &b1, &w2, &b2]).unwrap();
                        black_box(loss);
                    }
                    start.elapsed()
                });
            },
        );
    }

    group.finish();
}

/// Benchmark convolutional operations (simplified)
fn bench_convolution_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("convolution_comparison");

    let conv_configs = vec![
        (8, 3, 32, 32, 16),    // Small conv
        (16, 3, 64, 64, 32),   // Medium conv
        (32, 3, 128, 128, 64), // Large conv
    ];

    for (batch_size, channels, height, width, filters) in conv_configs {
        let config_name = format!(
            "conv_{}x{}x{}x{}_f{}",
            batch_size, channels, height, width, filters
        );

        group.bench_with_input(
            BenchmarkId::new("conv2d_forward_backward", &config_name),
            &(batch_size, channels, height, width, filters),
            |b, &(batch_size, channels, height, width, filters)| {
                let tape = GradientTape::new();

                // Create simplified convolution data (using matrix ops as approximation)
                let x_data = Array2::<f32>::zeros((batch_size, height * width)).into_dyn();
                let w_data = Array2::<f32>::zeros((filters, channels * 3 * 3)).into_dyn(); // 3x3 kernel
                let b_data = Array1::<f32>::zeros(filters).into_dyn();

                let x = tape.watch(Tensor::from_array(x_data));
                let w = tape.watch(Tensor::from_array(w_data));
                let b = tape.watch(Tensor::from_array(b_data));

                b.iter_custom(|iters| {
                    let start = Instant::now();
                    for _ in 0..iters {
                        // Simplified convolution using matrix operations
                        let conv_out = x.matmul(&w.transpose().unwrap()).unwrap().add(&b).unwrap();
                        let activated = conv_out.relu().unwrap();
                        let loss = activated.sum(None, false).unwrap();

                        let _grads = tape.gradient(&loss, &[&w, &b]).unwrap();
                        black_box(loss);
                    }
                    start.elapsed()
                });
            },
        );
    }

    group.finish();
}

/// Benchmark advanced mathematical operations
fn bench_advanced_math_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("advanced_math_comparison");

    let sizes = vec![100, 1000, 10000];

    for size in sizes {
        group.throughput(Throughput::Elements(size as u64));

        // Trigonometric functions
        group.bench_with_input(
            BenchmarkId::new("sin_cos_chain", size),
            &size,
            |b, &size| {
                let tape = GradientTape::new();
                let x_data = Array1::linspace(-3.14f32, 3.14, size).into_dyn();
                let x = tape.watch(Tensor::from_array(x_data));

                b.iter_custom(|iters| {
                    let start = Instant::now();
                    for _ in 0..iters {
                        // Chain of trig operations (using available ops as approximation)
                        let y = x.tanh().unwrap(); // Approximation for sin
                        let z = y.mul(&y).unwrap(); // Approximation for cos^2
                        let w = z.add(&y).unwrap();
                        let loss = w.sum(None, false).unwrap();

                        let _grads = tape.gradient(&loss, &[&x]).unwrap();
                        black_box(loss);
                    }
                    start.elapsed()
                });
            },
        );

        // Exponential and logarithmic operations
        group.bench_with_input(
            BenchmarkId::new("exp_log_chain", size),
            &size,
            |b, &size| {
                let tape = GradientTape::new();
                let x_data = Array1::linspace(0.1f32, 2.0, size).into_dyn();
                let x = tape.watch(Tensor::from_array(x_data));

                b.iter_custom(|iters| {
                    let start = Instant::now();
                    for _ in 0..iters {
                        // Chain using available operations as approximations
                        let y = x.sigmoid().unwrap(); // Has exp-like behavior
                        let z = y.add(&Tensor::from_scalar(1e-8)).unwrap(); // Add small epsilon
                        let w = z.pow(&Tensor::from_scalar(0.5)).unwrap(); // Approximation for sqrt/log
                        let loss = w.sum(None, false).unwrap();

                        let _grads = tape.gradient(&loss, &[&x]).unwrap();
                        black_box(loss);
                    }
                    start.elapsed()
                });
            },
        );
    }

    group.finish();
}

/// Benchmark higher-order derivatives
fn bench_higher_order_derivatives(c: &mut Criterion) {
    let mut group = c.benchmark_group("higher_order_derivatives");

    let sizes = vec![10, 100, 500]; // Smaller sizes for higher-order derivatives

    for size in sizes {
        group.bench_with_input(BenchmarkId::new("second_order", size), &size, |b, &size| {
            let tape1 = GradientTape::new();
            let x_data = Array1::linspace(0.1f32, 1.0, size).into_dyn();
            let x = tape1.watch(Tensor::from_array(x_data));

            b.iter_custom(|iters| {
                let start = Instant::now();
                for _ in 0..iters {
                    // First-order gradient
                    let y = x.pow(&Tensor::from_scalar(3.0)).unwrap();
                    let loss = y.sum(None, false).unwrap();
                    let first_grad = tape1.gradient(&loss, &[&x]).unwrap();

                    // Second-order gradient (simplified)
                    let tape2 = GradientTape::new();
                    let grad_tensor = tape2.watch(first_grad[0].clone());
                    let grad_loss = grad_tensor.sum(None, false).unwrap();
                    let _second_grad = tape2.gradient(&grad_loss, &[&grad_tensor]).unwrap();

                    black_box(loss);
                }
                start.elapsed()
            });
        });
    }

    group.finish();
}

/// Run external benchmark comparison (if Python scripts are available)
fn run_external_benchmark_comparison() -> Result<Vec<BenchmarkResult>, Box<dyn std::error::Error>> {
    let mut results = Vec::new();

    // Try to run PyTorch comparison
    if let Ok(output) = Command::new("python3")
        .arg("scripts/pytorch_benchmark.py")
        .arg("--output-json")
        .output()
    {
        if output.status.success() {
            let pytorch_results: Vec<BenchmarkResult> = serde_json::from_slice(&output.stdout)?;
            results.extend(pytorch_results);
        }
    }

    // Try to run TensorFlow comparison
    if let Ok(output) = Command::new("python3")
        .arg("scripts/tensorflow_benchmark.py")
        .arg("--output-json")
        .output()
    {
        if output.status.success() {
            let tf_results: Vec<BenchmarkResult> = serde_json::from_slice(&output.stdout)?;
            results.extend(tf_results);
        }
    }

    Ok(results)
}

/// Print comparison summary
fn print_benchmark_summary(results: &[BenchmarkResult]) {
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                         FRAMEWORK PERFORMANCE COMPARISON                        ║");
    println!("╠══════════════════════════════════════════════════════════════════════════════╣");

    let frameworks: std::collections::HashSet<String> =
        results.iter().map(|r| r.framework.clone()).collect();

    for framework in frameworks {
        println!("║ Framework: {:<65} ║", framework);

        let framework_results: Vec<_> = results
            .iter()
            .filter(|r| r.framework == framework)
            .collect();

        for result in framework_results {
            println!(
                "║   {:<20} | Size: {:>8} | Time: {:>8.2}ms | Throughput: {:>8.2} ops/s ║",
                result.operation,
                result.input_size,
                result.total_time_ms,
                result.throughput_ops_per_sec
            );
        }
        println!(
            "╠──────────────────────────────────────────────────────────────────────────────╣"
        );
    }

    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
}

/// Configuration for comprehensive benchmarks
fn configure_comprehensive_criterion() -> Criterion {
    Criterion::default()
        .sample_size(30)
        .measurement_time(Duration::from_secs(15))
        .warm_up_time(Duration::from_secs(3))
        .with_plots()
        .significance_level(0.05)
        .noise_threshold(0.02)
}

criterion_group!(
    name = pytorch_comparison_benches;
    config = configure_comprehensive_criterion();
    targets =
        bench_basic_operations_comparison,
        bench_matrix_operations_comparison,
        bench_neural_network_comparison,
        bench_convolution_comparison,
        bench_advanced_math_comparison,
        bench_higher_order_derivatives
);

criterion_main!(pytorch_comparison_benches);
