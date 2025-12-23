/// Dispatch Registry Benchmarks
///
/// Measures the overhead of the dispatch registry system compared to direct function calls.
/// These benchmarks help ensure that the unified dispatch system doesn't introduce
/// significant performance penalties for common operations.
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use scirs2_core::ndarray::{Array, Array1, Array2};
use tenflowers_core::{ensure_dispatch_initialized, Tensor, F32_REGISTRY};

/// Benchmark configuration for different tensor sizes
struct BenchConfig {
    name: &'static str,
    size: usize,
}

const SIZES: &[BenchConfig] = &[
    BenchConfig {
        name: "tiny_10",
        size: 10,
    },
    BenchConfig {
        name: "small_100",
        size: 100,
    },
    BenchConfig {
        name: "medium_1k",
        size: 1_000,
    },
    BenchConfig {
        name: "large_10k",
        size: 10_000,
    },
    BenchConfig {
        name: "xlarge_100k",
        size: 100_000,
    },
];

/// Direct CPU implementation of add (no dispatch)
fn add_direct_cpu(a: &Tensor<f32>, b: &Tensor<f32>) -> Tensor<f32> {
    let a_data = a.data();
    let b_data = b.data();
    let result: Vec<f32> = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(x, y)| x + y)
        .collect();
    let array = scirs2_core::ndarray::ArrayD::from_shape_vec(a.shape().dims(), result).unwrap();
    Tensor::from_array(array)
}

/// Direct CPU implementation of mul (no dispatch)
fn mul_direct_cpu(a: &Tensor<f32>, b: &Tensor<f32>) -> Tensor<f32> {
    let a_data = a.data();
    let b_data = b.data();
    let result: Vec<f32> = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(x, y)| x * y)
        .collect();
    let array = scirs2_core::ndarray::ArrayD::from_shape_vec(a.shape().dims(), result).unwrap();
    Tensor::from_array(array)
}

/// Direct CPU implementation of abs (no dispatch)
fn abs_direct_cpu(x: &Tensor<f32>) -> Tensor<f32> {
    let data = x.data();
    let result: Vec<f32> = data.iter().map(|v| v.abs()).collect();
    let array = scirs2_core::ndarray::ArrayD::from_shape_vec(x.shape().dims(), result).unwrap();
    Tensor::from_array(array)
}

/// Benchmark binary operations via dispatch registry
fn bench_dispatch_binary(c: &mut Criterion) {
    ensure_dispatch_initialized();

    let mut group = c.benchmark_group("dispatch_binary");

    for config in SIZES {
        let data_a: Vec<f32> = (0..config.size).map(|i| i as f32).collect();
        let data_b: Vec<f32> = (0..config.size).map(|i| (i as f32) * 2.0).collect();

        let a = Tensor::from_array(Array1::from_vec(data_a.clone()).into_dyn());
        let b = Tensor::from_array(Array1::from_vec(data_b.clone()).into_dyn());

        // Benchmark dispatch registry
        group.bench_with_input(
            BenchmarkId::new("add_dispatch", config.name),
            &(&a, &b),
            |bencher, (a, b)| {
                bencher.iter(|| {
                    F32_REGISTRY
                        .dispatch_binary("add", black_box(a), black_box(b))
                        .unwrap()
                });
            },
        );

        // Benchmark direct call
        group.bench_with_input(
            BenchmarkId::new("add_direct", config.name),
            &(&a, &b),
            |bencher, (a, b)| {
                bencher.iter(|| add_direct_cpu(black_box(a), black_box(b)));
            },
        );

        // Benchmark multiplication
        group.bench_with_input(
            BenchmarkId::new("mul_dispatch", config.name),
            &(&a, &b),
            |bencher, (a, b)| {
                bencher.iter(|| {
                    F32_REGISTRY
                        .dispatch_binary("mul", black_box(a), black_box(b))
                        .unwrap()
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("mul_direct", config.name),
            &(&a, &b),
            |bencher, (a, b)| {
                bencher.iter(|| mul_direct_cpu(black_box(a), black_box(b)));
            },
        );
    }

    group.finish();
}

/// Benchmark unary operations via dispatch registry
fn bench_dispatch_unary(c: &mut Criterion) {
    ensure_dispatch_initialized();

    let mut group = c.benchmark_group("dispatch_unary");

    for config in SIZES {
        let data: Vec<f32> = (0..config.size)
            .map(|i| (i as f32) - (config.size as f32 / 2.0))
            .collect();
        let tensor = Tensor::from_array(Array1::from_vec(data).into_dyn());

        // Benchmark dispatch registry
        group.bench_with_input(
            BenchmarkId::new("abs_dispatch", config.name),
            &tensor,
            |bencher, tensor| {
                bencher.iter(|| {
                    F32_REGISTRY
                        .dispatch_unary("abs", black_box(tensor))
                        .unwrap()
                });
            },
        );

        // Benchmark direct call
        group.bench_with_input(
            BenchmarkId::new("abs_direct", config.name),
            &tensor,
            |bencher, tensor| {
                bencher.iter(|| abs_direct_cpu(black_box(tensor)));
            },
        );
    }

    group.finish();
}

/// Benchmark dispatch overhead in isolation
fn bench_dispatch_overhead(c: &mut Criterion) {
    ensure_dispatch_initialized();

    let mut group = c.benchmark_group("dispatch_overhead");

    // Small tensor to isolate dispatch overhead
    let size = 10;
    let data_a: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let data_b: Vec<f32> = (0..size).map(|i| (i as f32) * 2.0).collect();

    let a = Tensor::from_array(Array1::from_vec(data_a).into_dyn());
    let b = Tensor::from_array(Array1::from_vec(data_b).into_dyn());

    // Measure pure dispatch overhead (registry lookup + backend selection)
    group.bench_function("registry_lookup", |bencher| {
        bencher.iter(|| {
            black_box(F32_REGISTRY.get_operation("add"));
        });
    });

    // Measure backend availability check
    group.bench_function("backend_check", |bencher| {
        bencher.iter(|| {
            black_box(F32_REGISTRY.available_backends("add"));
        });
    });

    // Full dispatch path
    group.bench_function("full_dispatch", |bencher| {
        bencher.iter(|| {
            F32_REGISTRY
                .dispatch_binary("add", black_box(&a), black_box(&b))
                .unwrap()
        });
    });

    // Direct call for comparison
    group.bench_function("direct_call", |bencher| {
        bencher.iter(|| add_direct_cpu(black_box(&a), black_box(&b)));
    });

    group.finish();
}

/// Benchmark 2D matrix operations
fn bench_dispatch_matrix(c: &mut Criterion) {
    ensure_dispatch_initialized();

    let mut group = c.benchmark_group("dispatch_matrix");

    let configs = vec![
        ("10x10", 10, 10),
        ("100x100", 100, 100),
        ("1000x1000", 1000, 1000),
    ];

    for (name, rows, cols) in configs {
        let size = rows * cols;
        let data_a: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let data_b: Vec<f32> = (0..size).map(|i| (i as f32) * 2.0).collect();

        let a = Tensor::from_array(
            Array2::from_shape_vec((rows, cols), data_a)
                .unwrap()
                .into_dyn(),
        );
        let b = Tensor::from_array(
            Array2::from_shape_vec((rows, cols), data_b)
                .unwrap()
                .into_dyn(),
        );

        group.bench_with_input(
            BenchmarkId::new("add_dispatch", name),
            &(&a, &b),
            |bencher, (a, b)| {
                bencher.iter(|| {
                    F32_REGISTRY
                        .dispatch_binary("add", black_box(a), black_box(b))
                        .unwrap()
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("add_direct", name),
            &(&a, &b),
            |bencher, (a, b)| {
                bencher.iter(|| add_direct_cpu(black_box(a), black_box(b)));
            },
        );
    }

    group.finish();
}

/// Benchmark chained operations
fn bench_dispatch_chained(c: &mut Criterion) {
    ensure_dispatch_initialized();

    let mut group = c.benchmark_group("dispatch_chained");

    let size = 1000;
    let data_a: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let data_b: Vec<f32> = (0..size).map(|i| (i as f32) * 2.0).collect();
    let data_c: Vec<f32> = (0..size).map(|i| (i as f32) + 1.0).collect();

    let a = Tensor::from_array(Array1::from_vec(data_a).into_dyn());
    let b = Tensor::from_array(Array1::from_vec(data_b).into_dyn());
    let c = Tensor::from_array(Array1::from_vec(data_c).into_dyn());

    // Benchmark (a + b) * c via dispatch
    group.bench_function("chained_dispatch", |bencher| {
        bencher.iter(|| {
            let temp = F32_REGISTRY
                .dispatch_binary("add", black_box(&a), black_box(&b))
                .unwrap();
            F32_REGISTRY
                .dispatch_binary("mul", black_box(&temp), black_box(&c))
                .unwrap()
        });
    });

    // Benchmark (a + b) * c direct
    group.bench_function("chained_direct", |bencher| {
        bencher.iter(|| {
            let temp = add_direct_cpu(black_box(&a), black_box(&b));
            mul_direct_cpu(black_box(&temp), black_box(&c))
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_dispatch_binary,
    bench_dispatch_unary,
    bench_dispatch_overhead,
    bench_dispatch_matrix,
    bench_dispatch_chained
);

criterion_main!(benches);
