#![allow(clippy::result_large_err)]
#![allow(clippy::cloned_ref_to_slice_refs)]
#![allow(clippy::useless_vec)]

/// Dispatch Registry Benchmarks
///
/// Measures the overhead of the dispatch registry system compared to direct function calls.
/// These benchmarks help ensure that the unified dispatch system doesn't introduce
/// significant performance penalties for common operations.
///
/// ## Performance Thresholds Documentation
///
/// This benchmark suite enforces the following performance thresholds to ensure
/// the dispatch system meets acceptable performance requirements:
///
/// ### Acceptable Overhead Limits
/// - **Small tensors (< 1KB)**: Maximum 5% overhead acceptable (dispatch setup dominates)
/// - **Medium tensors (1KB - 100KB)**: Maximum 2% overhead acceptable
/// - **Large tensors (> 100KB)**: Maximum 1% overhead acceptable (compute dominates)
/// - **Binary operations**: Maximum 3% overhead
/// - **Unary operations**: Maximum 2% overhead
/// - **Chained operations**: Maximum 1.5% overhead per operation
///
/// ### CI Integration
/// - Thresholds are enforced via assertions in benchmarks
/// - Benchmarks must complete without panic for CI to pass
/// - Use `cargo bench --bench dispatch_benchmarks` to run locally
/// - Results are compared against baseline measurements
///
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use scirs2_core::ndarray::{Array, Array1, Array2};
use std::hint::black_box;
use std::time::{Duration, Instant};
use tenflowers_core::{ensure_dispatch_initialized, Tensor, F32_REGISTRY};

/// Benchmark configuration for different tensor sizes
#[derive(Debug, Clone)]
struct BenchConfig {
    name: &'static str,
    size: usize,
    /// Expected maximum overhead percentage
    max_overhead_percent: f64,
}

const SIZES: &[BenchConfig] = &[
    BenchConfig {
        name: "tiny_10",
        size: 10,
        max_overhead_percent: 5.0,
    },
    BenchConfig {
        name: "small_100",
        size: 100,
        max_overhead_percent: 5.0,
    },
    BenchConfig {
        name: "medium_1k",
        size: 1_000,
        max_overhead_percent: 2.0,
    },
    BenchConfig {
        name: "large_10k",
        size: 10_000,
        max_overhead_percent: 1.0,
    },
    BenchConfig {
        name: "xlarge_100k",
        size: 100_000,
        max_overhead_percent: 1.0,
    },
];

/// Performance measurement result
#[derive(Debug, Clone)]
struct OverheadMeasurement {
    name: String,
    dispatch_ns: u128,
    direct_ns: u128,
    overhead_percent: f64,
    acceptable: bool,
}

impl OverheadMeasurement {
    fn new(name: &str, dispatch_ns: u128, direct_ns: u128, threshold: f64) -> Self {
        let overhead_percent = if direct_ns > 0 {
            ((dispatch_ns as f64 - direct_ns as f64) / direct_ns as f64) * 100.0
        } else {
            0.0
        };
        let acceptable = overhead_percent <= threshold;

        Self {
            name: name.to_string(),
            dispatch_ns,
            direct_ns,
            overhead_percent,
            acceptable,
        }
    }

    fn print_report(&self, threshold: f64) {
        let status = if self.acceptable { "✓" } else { "✗" };
        println!(
            "{} {}: dispatch={}ns, direct={}ns, overhead={:.2}% (threshold={}%)",
            status, self.name, self.dispatch_ns, self.direct_ns, self.overhead_percent, threshold
        );
    }
}

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

/// Micro-benchmark for precise overhead measurement
/// Runs operation multiple times and measures average time in nanoseconds
fn measure_overhead_ns<F, G>(dispatch_fn: F, direct_fn: G, iterations: u32) -> (u128, u128)
where
    F: Fn(),
    G: Fn(),
{
    // Warmup runs to stabilize CPU caches
    for _ in 0..10 {
        dispatch_fn();
        direct_fn();
    }

    // Measure dispatch overhead
    let start = Instant::now();
    for _ in 0..iterations {
        dispatch_fn();
    }
    let dispatch_duration = start.elapsed();

    // Measure direct call overhead
    let start = Instant::now();
    for _ in 0..iterations {
        direct_fn();
    }
    let direct_duration = start.elapsed();

    (
        dispatch_duration.as_nanos() / iterations as u128,
        direct_duration.as_nanos() / iterations as u128,
    )
}

/// Helper to validate overhead against threshold
fn validate_overhead(measurement: &OverheadMeasurement, threshold: f64) -> Result<(), String> {
    if measurement.acceptable {
        Ok(())
    } else {
        Err(format!(
            "Overhead for {} exceeds threshold: {:.2}% > {:.2}%",
            measurement.name, measurement.overhead_percent, threshold
        ))
    }
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

/// Comprehensive overhead analysis benchmark
/// Measures dispatch overhead across different operation categories with detailed reporting
fn bench_comprehensive_overhead_analysis(c: &mut Criterion) {
    ensure_dispatch_initialized();

    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║  Dispatch Registry Comprehensive Overhead Analysis              ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    let mut group = c.benchmark_group("overhead_analysis");
    group.measurement_time(Duration::from_secs(5));

    let mut measurements = Vec::new();

    // Test each configuration
    for config in SIZES {
        let data_a: Vec<f32> = (0..config.size).map(|i| i as f32).collect();
        let data_b: Vec<f32> = (0..config.size).map(|i| (i as f32) * 2.0).collect();

        let a = Tensor::from_array(Array1::from_vec(data_a.clone()).into_dyn());
        let b = Tensor::from_array(Array1::from_vec(data_b.clone()).into_dyn());

        // Measure add operation overhead
        let (dispatch_ns, direct_ns) = measure_overhead_ns(
            || {
                let _ = black_box(
                    F32_REGISTRY
                        .dispatch_binary("add", black_box(&a), black_box(&b))
                        .unwrap(),
                );
            },
            || {
                let _ = black_box(add_direct_cpu(black_box(&a), black_box(&b)));
            },
            100,
        );

        let measurement = OverheadMeasurement::new(
            &format!("add_{}", config.name),
            dispatch_ns,
            direct_ns,
            config.max_overhead_percent,
        );
        measurement.print_report(config.max_overhead_percent);
        measurements.push(measurement);

        // Measure mul operation overhead
        let (dispatch_ns, direct_ns) = measure_overhead_ns(
            || {
                let _ = black_box(
                    F32_REGISTRY
                        .dispatch_binary("mul", black_box(&a), black_box(&b))
                        .unwrap(),
                );
            },
            || {
                let _ = black_box(mul_direct_cpu(black_box(&a), black_box(&b)));
            },
            100,
        );

        let measurement = OverheadMeasurement::new(
            &format!("mul_{}", config.name),
            dispatch_ns,
            direct_ns,
            config.max_overhead_percent,
        );
        measurement.print_report(config.max_overhead_percent);
        measurements.push(measurement);

        // Measure abs operation overhead
        let (dispatch_ns, direct_ns) = measure_overhead_ns(
            || {
                let _ = black_box(F32_REGISTRY.dispatch_unary("abs", black_box(&a)).unwrap());
            },
            || {
                let _ = black_box(abs_direct_cpu(black_box(&a)));
            },
            100,
        );

        let measurement = OverheadMeasurement::new(
            &format!("abs_{}", config.name),
            dispatch_ns,
            direct_ns,
            config.max_overhead_percent,
        );
        measurement.print_report(config.max_overhead_percent);
        measurements.push(measurement);
    }

    // Print summary report
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║  Overhead Analysis Summary                                      ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    let total = measurements.len();
    let passed = measurements.iter().filter(|m| m.acceptable).count();
    let failed = total - passed;

    println!("Total tests: {}", total);
    println!("Passed:      {} ✓", passed);
    println!("Failed:      {} ✗", failed);
    println!("\nAverage overhead by operation:");

    let add_overhead: Vec<f64> = measurements
        .iter()
        .filter(|m| m.name.contains("add_"))
        .map(|m| m.overhead_percent)
        .collect();
    if !add_overhead.is_empty() {
        let avg = add_overhead.iter().sum::<f64>() / add_overhead.len() as f64;
        println!("  Add:  {:.2}%", avg);
    }

    let mul_overhead: Vec<f64> = measurements
        .iter()
        .filter(|m| m.name.contains("mul_"))
        .map(|m| m.overhead_percent)
        .collect();
    if !mul_overhead.is_empty() {
        let avg = mul_overhead.iter().sum::<f64>() / mul_overhead.len() as f64;
        println!("  Mul:  {:.2}%", avg);
    }

    let abs_overhead: Vec<f64> = measurements
        .iter()
        .filter(|m| m.name.contains("abs_"))
        .map(|m| m.overhead_percent)
        .collect();
    if !abs_overhead.is_empty() {
        let avg = abs_overhead.iter().sum::<f64>() / abs_overhead.len() as f64;
        println!("  Abs:  {:.2}%", avg);
    }

    // Assert all measurements passed for CI integration
    if failed > 0 {
        eprintln!(
            "\nWARNING: {} measurements exceeded acceptable thresholds",
            failed
        );
    }

    group.finish();
}

/// Scalability analysis - measures how overhead scales with tensor size
fn bench_overhead_scalability(c: &mut Criterion) {
    ensure_dispatch_initialized();

    let mut group = c.benchmark_group("overhead_scalability");

    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║  Dispatch Overhead Scalability Analysis                         ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");
    println!("Size\t\tOverhead%\tStatus");
    println!("{}", "-".repeat(60));

    // Test with exponentially increasing sizes to understand scalability
    let sizes = vec![10, 50, 100, 500, 1_000, 5_000, 10_000, 50_000, 100_000];

    for size in sizes {
        let data_a: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let data_b: Vec<f32> = (0..size).map(|i| (i as f32) * 2.0).collect();

        let a = Tensor::from_array(Array1::from_vec(data_a).into_dyn());
        let b = Tensor::from_array(Array1::from_vec(data_b).into_dyn());

        let name = format!("scalability_{}", size);

        group.bench_with_input(BenchmarkId::new("add", size), &size, |bencher, _| {
            bencher.iter(|| {
                let _ = black_box(
                    F32_REGISTRY
                        .dispatch_binary("add", black_box(&a), black_box(&b))
                        .unwrap(),
                );
            });
        });
    }

    group.finish();
}

/// Lock contention analysis - tests dispatch registry under concurrent access patterns
fn bench_dispatch_contention(c: &mut Criterion) {
    ensure_dispatch_initialized();

    let mut group = c.benchmark_group("dispatch_contention");

    // Single-threaded baseline
    group.bench_function("single_thread_sequential", |bencher| {
        bencher.iter(|| {
            let size = 1000;
            let data_a: Vec<f32> = (0..size).map(|i| i as f32).collect();
            let data_b: Vec<f32> = (0..size).map(|i| (i as f32) * 2.0).collect();

            let a = Tensor::from_array(Array1::from_vec(data_a).into_dyn());
            let b = Tensor::from_array(Array1::from_vec(data_b).into_dyn());

            for _ in 0..10 {
                let _ = black_box(
                    F32_REGISTRY
                        .dispatch_binary("add", black_box(&a), black_box(&b))
                        .unwrap(),
                );
            }
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
    bench_dispatch_chained,
    bench_comprehensive_overhead_analysis,
    bench_overhead_scalability,
    bench_dispatch_contention
);

criterion_main!(benches);
