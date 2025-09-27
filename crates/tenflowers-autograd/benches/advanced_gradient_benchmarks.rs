use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_autograd::ndarray::{Array1, Array2, Array3, Array4};
use std::time::Duration;
use tenflowers_autograd::{get_global_profiler, GradientTape, MemoryStats};
use tenflowers_core::{DType, Device, Tensor};

/// Benchmark edge cases and extreme scenarios for gradient computation
fn bench_gradient_edge_cases(c: &mut Criterion) {
    let mut group = c.benchmark_group("gradient_edge_cases");
    group.sample_size(50); // Reduced sample size for potentially slow operations

    // Test with very small tensors (single elements)
    group.bench_function("scalar_gradients", |b| {
        let tape = GradientTape::new();
        let x = tape.watch(Tensor::from_array(
            Array1::from_vec(vec![2.0f32]).into_dyn(),
        ));
        let y = tape.watch(Tensor::from_array(
            Array1::from_vec(vec![3.0f32]).into_dyn(),
        ));

        b.iter(|| {
            let z = x.mul(&y).unwrap();
            let pow_z = z.pow(&x).unwrap();
            black_box(tape.gradient(&pow_z, &[&x, &y]).unwrap());
        });
    });

    // Test with very large 1D tensors
    let large_size = 100_000;
    group.throughput(Throughput::Elements(large_size as u64));
    group.bench_function("large_1d_chain", |b| {
        let tape = GradientTape::new();
        let x_data = Array1::linspace(0.1f32, 1.0, large_size).into_dyn();
        let x = tape.watch(Tensor::from_array(x_data));

        b.iter(|| {
            let y = x.relu().unwrap();
            let z = y.sigmoid().unwrap();
            let w = z.tanh().unwrap();
            black_box(tape.gradient(&w, &[&x]).unwrap());
        });
    });

    // Test with extreme aspect ratios
    group.bench_function("extreme_aspect_ratio", |b| {
        let tape = GradientTape::new();
        let narrow_data = Array2::<f32>::ones((10000, 1)).into_dyn();
        let wide_data = Array2::<f32>::ones((1, 10000)).into_dyn();
        let x = tape.watch(Tensor::from_array(narrow_data));
        let y = tape.watch(Tensor::from_array(wide_data));

        b.iter(|| {
            let z = x.matmul(&y).unwrap();
            black_box(tape.gradient(&z, &[&x, &y]).unwrap());
        });
    });

    // Test with high-dimensional tensors
    group.bench_function("high_dimension_tensors", |b| {
        let tape = GradientTape::new();
        let tensor_data = Array4::<f32>::ones((10, 5, 4, 3)).into_dyn();
        let x = tape.watch(Tensor::from_array(tensor_data.clone()));
        let y = tape.watch(Tensor::from_array(tensor_data));

        b.iter(|| {
            let z = x.add(&y).unwrap();
            let w = z.sum(None, false).unwrap();
            black_box(tape.gradient(&w, &[&x, &y]).unwrap());
        });
    });

    group.finish();
}

/// Benchmark numerical stability edge cases
fn bench_numerical_stability(c: &mut Criterion) {
    let mut group = c.benchmark_group("numerical_stability");
    group.sample_size(30);

    // Test with very small values (near zero)
    group.bench_function("near_zero_values", |b| {
        let tape = GradientTape::new();
        let small_data = Array1::from_vec(vec![1e-10f32, 1e-9f32, 1e-8f32]).into_dyn();
        let x = tape.watch(Tensor::from_array(small_data));

        b.iter(|| {
            let y = x.sigmoid().unwrap(); // Using sigmoid for numerical stability
            let z = y.relu().unwrap();
            black_box(tape.gradient(&z, &[&x]).unwrap());
        });
    });

    // Test with very large values
    group.bench_function("large_values", |b| {
        let tape = GradientTape::new();
        let large_data = Array1::from_vec(vec![1e6f32, 1e7f32, 1e8f32]).into_dyn();
        let x = tape.watch(Tensor::from_array(large_data));

        b.iter(|| {
            let y = x.sigmoid().unwrap();
            let z = y.tanh().unwrap(); // Using tanh instead of log for numerical stability
            black_box(tape.gradient(&z, &[&x]).unwrap());
        });
    });

    // Test with mixed positive/negative values
    group.bench_function("mixed_sign_values", |b| {
        let tape = GradientTape::new();
        let mixed_data = Array1::from_vec(vec![-100.0f32, 0.0f32, 100.0f32]).into_dyn();
        let x = tape.watch(Tensor::from_array(mixed_data));

        b.iter(|| {
            let y = x.tanh().unwrap();
            let z = y.pow(&x).unwrap();
            black_box(tape.gradient(&z, &[&x]).unwrap());
        });
    });

    group.finish();
}

/// Benchmark complex gradient computation chains
fn bench_complex_gradient_chains(c: &mut Criterion) {
    let mut group = c.benchmark_group("complex_gradient_chains");

    // Deep computation chain (many operations)
    group.bench_function("deep_chain", |b| {
        let tape = GradientTape::new();
        let x_data = Array1::from_vec(vec![1.0f32, 2.0f32, 3.0f32]).into_dyn();
        let x = tape.watch(Tensor::from_array(x_data));

        b.iter(|| {
            let mut result = x.clone();
            // Create a deep computation chain
            for i in 1..20 {
                let scalar = tape.watch(Tensor::from_array(
                    Array1::from_vec(vec![i as f32 * 0.1]).into_dyn(),
                ));
                result = result.mul(&scalar).unwrap();
                result = result.sigmoid().unwrap();
                result = result.add(&x).unwrap();
            }
            black_box(tape.gradient(&result, &[&x]).unwrap());
        });
    });

    // Wide computation graph (many parallel branches)
    group.bench_function("wide_graph", |b| {
        let tape = GradientTape::new();
        let x_data = Array1::from_vec(vec![1.0f32, 2.0f32, 3.0f32]).into_dyn();
        let x = tape.watch(Tensor::from_array(x_data));

        b.iter(|| {
            // Create multiple parallel branches
            let branch1 = x.sigmoid().unwrap();
            let branch2 = x.tanh().unwrap();
            let branch3 = x.relu().unwrap();
            let branch4 = x.pow(&x).unwrap();
            let branch5 = x.sigmoid().unwrap(); // Using sigmoid for numerical stability

            // Combine branches
            let combined = branch1
                .add(&branch2)
                .unwrap()
                .add(&branch3)
                .unwrap()
                .add(&branch4)
                .unwrap()
                .add(&branch5)
                .unwrap();

            black_box(tape.gradient(&combined, &[&x]).unwrap());
        });
    });

    // Mixed tensor shapes in computation
    group.bench_function("mixed_shapes", |b| {
        let tape = GradientTape::new();
        let vec_data = Array1::from_vec(vec![1.0f32, 2.0f32, 3.0f32]).into_dyn();
        let matrix_data =
            Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
                .unwrap()
                .into_dyn();

        let x = tape.watch(Tensor::from_array(vec_data));
        let m = tape.watch(Tensor::from_array(matrix_data));

        b.iter(|| {
            let y = m.matmul(&x.reshape(&[3, 1]).unwrap()).unwrap();
            let z = y.sigmoid().unwrap();
            let w = z.sum(None, false).unwrap();
            black_box(tape.gradient(&w, &[&x, &m]).unwrap());
        });
    });

    group.finish();
}

/// Benchmark memory usage patterns during gradient computation
fn bench_memory_intensive_gradients(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_intensive");
    group.sample_size(20); // Fewer samples for memory-intensive tests

    // Test memory profiler integration
    group.bench_function("memory_profiled_computation", |b| {
        let tape = GradientTape::new();
        let data = Array2::<f32>::ones((1000, 1000)).into_dyn();
        let x = tape.watch(Tensor::from_array(data.clone()));
        let y = tape.watch(Tensor::from_array(data));

        b.iter(|| {
            let profiler = get_global_profiler();
            if let Ok(mut p) = profiler.lock() {
                p.reset_stats().unwrap();
                p.begin_operation("large_matrix_ops").unwrap();
            }

            let z = x.matmul(&y).unwrap();
            let w = z.sigmoid().unwrap();
            let result = tape.gradient(&w, &[&x, &y]).unwrap();

            if let Ok(mut p) = profiler.lock() {
                p.end_operation().unwrap();
                let stats = p.get_stats().unwrap();
                // Record peak memory usage as part of benchmark
                black_box(stats.peak_memory);
            }

            black_box(result);
        });
    });

    // Test gradient computation with memory pressure
    group.bench_function("memory_pressure", |b| {
        let tape = GradientTape::new();
        // Create many intermediate tensors to stress memory
        let base_data = Array2::<f32>::ones((500, 500)).into_dyn();
        let tensors: Vec<_> = (0..10)
            .map(|_| tape.watch(Tensor::from_array(base_data.clone())))
            .collect();

        b.iter(|| {
            let mut result = tensors[0].clone();
            for tensor in &tensors[1..] {
                result = result.add(tensor).unwrap();
                result = result.sigmoid().unwrap();
            }
            let tensor_refs: Vec<&_> = tensors.iter().collect();
            black_box(tape.gradient(&result, &tensor_refs).unwrap());
        });
    });

    group.finish();
}

/// Benchmark gradient computation with different data types and precision
fn bench_precision_variants(c: &mut Criterion) {
    let mut group = c.benchmark_group("precision_variants");

    // f64 precision gradients
    group.bench_function("f64_precision", |b| {
        let tape = GradientTape::new();
        let x_data = Array1::from_vec(vec![1.0f64, 2.0f64, 3.0f64]).into_dyn();
        let y_data = Array1::from_vec(vec![4.0f64, 5.0f64, 6.0f64]).into_dyn();
        let x = tape.watch(Tensor::from_array(x_data));
        let y = tape.watch(Tensor::from_array(y_data));

        b.iter(|| {
            let z = x.mul(&y).unwrap();
            let w = z.pow(&x).unwrap();
            let result = w.sum(None, false).unwrap();
            black_box(tape.gradient(&result, &[&x, &y]).unwrap());
        });
    });

    // f32 precision gradients (for comparison)
    group.bench_function("f32_precision", |b| {
        let tape = GradientTape::new();
        let x_data = Array1::from_vec(vec![1.0f32, 2.0f32, 3.0f32]).into_dyn();
        let y_data = Array1::from_vec(vec![4.0f32, 5.0f32, 6.0f32]).into_dyn();
        let x = tape.watch(Tensor::from_array(x_data));
        let y = tape.watch(Tensor::from_array(y_data));

        b.iter(|| {
            let z = x.mul(&y).unwrap();
            let w = z.pow(&x).unwrap();
            let result = w.sum(None, false).unwrap();
            black_box(tape.gradient(&result, &[&x, &y]).unwrap());
        });
    });

    group.finish();
}

/// Benchmark error handling and recovery scenarios
fn bench_error_scenarios(c: &mut Criterion) {
    let mut group = c.benchmark_group("error_scenarios");
    group.sample_size(100);

    // Test gradient computation with shape mismatches
    group.bench_function("shape_mismatch_recovery", |b| {
        let tape = GradientTape::new();
        let x_data = Array1::from_vec(vec![1.0f32, 2.0f32]).into_dyn();
        let y_data = Array1::from_vec(vec![3.0f32, 4.0f32, 5.0f32]).into_dyn();
        let x = tape.watch(Tensor::from_array(x_data));
        let y = tape.watch(Tensor::from_array(y_data));

        b.iter(|| {
            // This should fail, but we want to benchmark error handling
            let result = x.matmul(&y);
            if result.is_ok() {
                // Unexpected success, compute gradients
                black_box(tape.gradient(&result.unwrap(), &[&x, &y]));
            } else {
                // Expected failure, benchmark error path
                black_box(result.unwrap_err());
            }
        });
    });

    group.finish();
}

/// Comprehensive benchmark suite covering all edge cases
fn bench_comprehensive_suite(c: &mut Criterion) {
    bench_gradient_edge_cases(c);
    bench_numerical_stability(c);
    bench_complex_gradient_chains(c);
    bench_memory_intensive_gradients(c);
    bench_precision_variants(c);
    bench_error_scenarios(c);
}

criterion_group!(benches, bench_comprehensive_suite);
criterion_main!(benches);
