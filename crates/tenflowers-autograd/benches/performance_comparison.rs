use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray::{Array1, Array2};
use std::time::{Duration, Instant};
use tenflowers_autograd::{GradientAccumulator, GradientTape};
use tenflowers_core::{DType, Device, Tensor};

/// Benchmark suite for comparing TenfloweRS with reference implementations
/// This provides performance metrics that can be compared with PyTorch/JAX/TensorFlow
/// Note: This is a framework for comparison - actual reference implementations would need to be added

/// Performance regression tests to ensure gradient computation doesn't degrade over time
fn bench_performance_regression(c: &mut Criterion) {
    let mut group = c.benchmark_group("performance_regression");

    // Set baseline expectations (these should be updated as performance improves)
    let baseline_expectations = vec![
        ("basic_add_1000", 1000.0), // Expected ops per second
        ("basic_mul_1000", 1000.0),
        ("relu_10000", 10000.0),
        ("sigmoid_10000", 5000.0),
        ("matmul_100x100", 100.0),
        ("neural_network_forward_backward", 50.0),
    ];

    // Basic operations regression tests
    group.bench_function("basic_add_1000", |b| {
        let tape = GradientTape::new();
        let x_data = Array1::linspace(0.0f32, 1.0, 1000).into_dyn();
        let y_data = Array1::linspace(1.0f32, 2.0, 1000).into_dyn();
        let x = tape.watch(Tensor::from_array(x_data));
        let y = tape.watch(Tensor::from_array(y_data));

        b.iter(|| {
            let z = x.add(&y).unwrap();
            black_box(tape.gradient(&z, &[&x, &y]).unwrap());
        });
    });

    group.bench_function("basic_mul_1000", |b| {
        let tape = GradientTape::new();
        let x_data = Array1::linspace(0.1f32, 1.0, 1000).into_dyn();
        let y_data = Array1::linspace(1.0f32, 2.0, 1000).into_dyn();
        let x = tape.watch(Tensor::from_array(x_data));
        let y = tape.watch(Tensor::from_array(y_data));

        b.iter(|| {
            let z = x.mul(&y).unwrap();
            black_box(tape.gradient(&z, &[&x, &y]).unwrap());
        });
    });

    group.bench_function("relu_10000", |b| {
        let tape = GradientTape::new();
        let x_data = Array1::linspace(-1.0f32, 1.0, 10000).into_dyn();
        let x = tape.watch(Tensor::from_array(x_data));

        b.iter(|| {
            let z = x.relu().unwrap();
            black_box(tape.gradient(&z, &[&x]).unwrap());
        });
    });

    group.bench_function("sigmoid_10000", |b| {
        let tape = GradientTape::new();
        let x_data = Array1::linspace(-5.0f32, 5.0, 10000).into_dyn();
        let x = tape.watch(Tensor::from_array(x_data));

        b.iter(|| {
            let z = x.sigmoid().unwrap();
            black_box(tape.gradient(&z, &[&x]).unwrap());
        });
    });

    group.bench_function("matmul_100x100", |b| {
        let tape = GradientTape::new();
        let x_data = Array2::<f32>::zeros((100, 100)).into_dyn();
        let y_data = Array2::<f32>::eye(100).into_dyn();
        let x = tape.watch(Tensor::from_array(x_data));
        let y = tape.watch(Tensor::from_array(y_data));

        b.iter(|| {
            let z = x.matmul(&y).unwrap();
            black_box(tape.gradient(&z, &[&x, &y]).unwrap());
        });
    });

    group.bench_function("neural_network_forward_backward", |b| {
        let tape = GradientTape::new();
        let batch_size = 32;
        let input_size = 784;
        let hidden_size = 128;
        let output_size = 10;

        // Create input data
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

        b.iter(|| {
            // Forward pass
            let h1 = x.matmul(&w1).unwrap().add(&b1).unwrap();
            let h1_relu = h1.relu().unwrap();
            let h2 = h1_relu.matmul(&w2).unwrap().add(&b2).unwrap();
            let output = h2.softmax(Some(1)).unwrap();
            let loss = output.sum(None, false).unwrap();

            // Backward pass
            black_box(tape.gradient(&loss, &[&w1, &b1, &w2, &b2]).unwrap());
        });
    });

    group.finish();
}

/// Benchmark memory usage during gradient computation
fn bench_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");

    // Test memory usage patterns for different tensor sizes
    let sizes = vec![100, 1000, 10000];

    for size in sizes {
        group.bench_with_input(
            BenchmarkId::new("memory_overhead", size),
            &size,
            |b, &size| {
                b.iter_custom(|iters| {
                    let start = Instant::now();

                    for _ in 0..iters {
                        let tape = GradientTape::new();
                        let x_data = Array1::zeros(size).into_dyn();
                        let x = tape.watch(Tensor::from_array(x_data));

                        // Perform operations that create intermediate tensors
                        let y = x.add(&x).unwrap();
                        let z = y.mul(&y).unwrap();
                        let w = z.relu().unwrap();

                        // Compute gradients
                        let _gradients = tape.gradient(&w, &[&x]).unwrap();

                        // Drop everything to simulate memory cleanup
                        drop(tape);
                        drop(x);
                        drop(y);
                        drop(z);
                        drop(w);
                    }

                    start.elapsed()
                });
            },
        );
    }

    group.finish();
}

/// Benchmark gradient computation overhead compared to forward pass
fn bench_gradient_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("gradient_overhead");

    // Test the overhead of gradient computation vs forward pass only
    let size = 1000;

    // Forward pass only
    group.bench_function("forward_only", |b| {
        let x_data = Array1::linspace(0.0f32, 1.0, size).into_dyn();
        let y_data = Array1::linspace(1.0f32, 2.0, size).into_dyn();
        let x = Tensor::from_array(x_data);
        let y = Tensor::from_array(y_data);

        b.iter(|| {
            let z = x.add(&y).unwrap();
            let w = z.relu().unwrap();
            black_box(w.sum(None, false).unwrap());
        });
    });

    // Forward + backward pass
    group.bench_function("forward_backward", |b| {
        let tape = GradientTape::new();
        let x_data = Array1::linspace(0.0f32, 1.0, size).into_dyn();
        let y_data = Array1::linspace(1.0f32, 2.0, size).into_dyn();
        let x = tape.watch(Tensor::from_array(x_data));
        let y = tape.watch(Tensor::from_array(y_data));

        b.iter(|| {
            let z = x.add(&y).unwrap();
            let w = z.relu().unwrap();
            let loss = w.sum(None, false).unwrap();
            black_box(tape.gradient(&loss, &[&x, &y]).unwrap());
        });
    });

    group.finish();
}

/// Benchmark scalability with different batch sizes
fn bench_scalability(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalability");

    let batch_sizes = vec![1, 8, 32, 128];
    let feature_size = 784;

    for batch_size in batch_sizes {
        group.throughput(Throughput::Elements((batch_size * feature_size) as u64));

        group.bench_with_input(
            BenchmarkId::new("batch_gradient", batch_size),
            &batch_size,
            |b, &batch_size| {
                let tape = GradientTape::new();
                let x_data = Array2::zeros((batch_size, feature_size)).into_dyn();
                let w_data = Array2::zeros((feature_size, 10)).into_dyn();
                let b_data = Array1::zeros(10).into_dyn();

                let x = tape.watch(Tensor::from_array(x_data));
                let w = tape.watch(Tensor::from_array(w_data));
                let b = tape.watch(Tensor::from_array(b_data));

                b.iter(|| {
                    let logits = x.matmul(&w).unwrap().add(&b).unwrap();
                    let probs = logits.softmax(Some(1)).unwrap();
                    let loss = probs.sum(None, false).unwrap();
                    black_box(tape.gradient(&loss, &[&w, &b]).unwrap());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark different gradient accumulation strategies
fn bench_gradient_accumulation_strategies(c: &mut Criterion) {
    let mut group = c.benchmark_group("gradient_accumulation_strategies");

    let total_batch_size = 128;
    let micro_batch_sizes = vec![1, 4, 16, 32];

    for micro_batch_size in micro_batch_sizes {
        if micro_batch_size <= total_batch_size {
            group.bench_with_input(
                BenchmarkId::new("micro_batch", micro_batch_size),
                &micro_batch_size,
                |b, &micro_batch_size| {
                    let tape = GradientTape::new();
                    let accumulator = GradientAccumulator::new(true);

                    let feature_size = 100;
                    let x_data = Array2::zeros((micro_batch_size, feature_size)).into_dyn();
                    let w_data = Array2::zeros((feature_size, 1)).into_dyn();

                    let x = tape.watch(Tensor::from_array(x_data));
                    let w = tape.watch(Tensor::from_array(w_data));

                    let num_micro_batches = total_batch_size / micro_batch_size;

                    b.iter(|| {
                        accumulator.clear();

                        // Accumulate gradients across micro-batches
                        for _ in 0..num_micro_batches {
                            let logits = x.matmul(&w).unwrap();
                            let loss = logits.sum(None, false).unwrap();
                            accumulator.accumulate(&tape, &loss, &[&w]).unwrap();
                        }

                        // Get accumulated gradients
                        black_box(accumulator.get_gradient(&w).unwrap());
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark complex operation chains
fn bench_complex_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("complex_operations");

    // Test different chain lengths
    let chain_lengths = vec![5, 10, 20, 50];

    for chain_length in chain_lengths {
        group.bench_with_input(
            BenchmarkId::new("operation_chain", chain_length),
            &chain_length,
            |b, &chain_length| {
                let tape = GradientTape::new();
                let x_data = Array1::ones(100).into_dyn();
                let x = tape.watch(Tensor::from_array(x_data));

                b.iter(|| {
                    let mut result = x.clone();

                    // Create a chain of operations
                    for i in 0..chain_length {
                        match i % 4 {
                            0 => result = result.add(&result).unwrap(),
                            1 => {
                                let scalar = tape.watch(Tensor::from_scalar(0.5));
                                result = result.mul(&scalar).unwrap();
                            }
                            2 => result = result.relu().unwrap(),
                            3 => result = result.tanh().unwrap(),
                            _ => unreachable!(),
                        }
                    }

                    black_box(tape.gradient(&result, &[&x]).unwrap());
                });
            },
        );
    }

    group.finish();
}

/// Performance testing utilities
pub struct PerformanceReport {
    pub operation: String,
    pub input_size: usize,
    pub time_per_op: Duration,
    pub throughput: f64,
    pub memory_usage: Option<usize>,
}

impl PerformanceReport {
    pub fn new(
        operation: String,
        input_size: usize,
        time_per_op: Duration,
        throughput: f64,
    ) -> Self {
        Self {
            operation,
            input_size,
            time_per_op,
            throughput,
            memory_usage: None,
        }
    }

    pub fn with_memory_usage(mut self, memory_usage: usize) -> Self {
        self.memory_usage = Some(memory_usage);
        self
    }

    pub fn print_report(&self) {
        println!("Operation: {}", self.operation);
        println!("Input size: {}", self.input_size);
        println!("Time per operation: {:?}", self.time_per_op);
        println!("Throughput: {:.2} ops/sec", self.throughput);
        if let Some(memory) = self.memory_usage {
            println!("Memory usage: {} bytes", memory);
        }
        println!("---");
    }
}

/// Configuration for performance benchmarks
fn configure_performance_criterion() -> Criterion {
    Criterion::default()
        .sample_size(50)
        .measurement_time(Duration::from_secs(10))
        .warm_up_time(Duration::from_secs(2))
        .with_plots()
        .significance_level(0.05)
        .noise_threshold(0.02)
}

criterion_group!(
    name = performance_benches;
    config = configure_performance_criterion();
    targets =
        bench_performance_regression,
        bench_memory_usage,
        bench_gradient_overhead,
        bench_scalability,
        bench_gradient_accumulation_strategies,
        bench_complex_operations
);

criterion_main!(performance_benches);
