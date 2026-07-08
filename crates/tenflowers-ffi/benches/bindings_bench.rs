//! Performance benchmarks for the tenflowers-ffi Python bindings layer.
//!
//! These benchmarks measure the overhead of the Rust-side of the FFI layer
//! without requiring a Python interpreter.  They cover:
//!
//! - Gradient parity checking throughput (central vs. forward differences)
//! - Tensor creation overhead (`zeros`, `ones`) at various sizes
//! - Basic element-wise arithmetic throughput (`add`, `mul`)
//! - Numeric Jacobian computation
//! - A head-to-head comparison of finite-difference methods

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use tenflowers::gradient_parity::{numeric_jacobian, GradientParityChecker};
use tenflowers_core::Tensor;

// ─── Gradient parity throughput ───────────────────────────────────────────────

/// Benchmark gradient parity checking across several input sizes.
fn bench_gradient_parity(c: &mut Criterion) {
    let checker = GradientParityChecker::new(1e-5, 1e-4);

    let mut group = c.benchmark_group("gradient_parity");

    for &size in &[16usize, 64, 256, 1024] {
        let input: Vec<f32> = (0..size).map(|i| i as f32 * 0.1 + 0.1).collect();
        let analytical_grad: Vec<f32> = input.iter().map(|&x| 2.0 * x).collect();

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("check_quadratic", size), &size, |b, _| {
            b.iter(|| {
                checker
                    .check_scalar_function(
                        |x| x.iter().map(|&v| v * v).sum::<f32>(),
                        black_box(&input),
                        black_box(&analytical_grad),
                    )
                    .expect("bench: parity check must succeed")
            });
        });

        group.bench_with_input(BenchmarkId::new("numeric_jacobian", size), &size, |b, _| {
            b.iter(|| {
                numeric_jacobian(|x| x.to_vec(), black_box(&input), size, 1e-3, true)
                    .expect("bench: jacobian must succeed")
            });
        });
    }

    group.finish();
}

// ─── Tensor creation overhead ─────────────────────────────────────────────────

/// Benchmark `Tensor::zeros` and `Tensor::ones` at several sizes.
fn bench_tensor_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_creation");

    for &size in &[64usize, 256, 1024, 4096] {
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("zeros", size), &size, |b, &s| {
            b.iter(|| black_box(Tensor::<f32>::zeros(&[s])));
        });

        group.bench_with_input(BenchmarkId::new("ones", size), &size, |b, &s| {
            b.iter(|| black_box(Tensor::<f32>::ones(&[s])));
        });
    }

    group.finish();
}

// ─── Arithmetic throughput ────────────────────────────────────────────────────

/// Benchmark element-wise `add` and `mul` at several sizes.
fn bench_arithmetic(c: &mut Criterion) {
    let mut group = c.benchmark_group("arithmetic");

    for &size in &[256usize, 1024, 4096, 16384] {
        let a = Tensor::<f32>::ones(&[size]);
        let b = Tensor::<f32>::ones(&[size]);

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("add", size), &size, |b_bench, _| {
            b_bench.iter(|| {
                tenflowers_core::ops::add(black_box(&a), black_box(&b))
                    .expect("bench: add must succeed")
            });
        });

        group.bench_with_input(BenchmarkId::new("mul", size), &size, |b_bench, _| {
            b_bench.iter(|| {
                tenflowers_core::ops::mul(black_box(&a), black_box(&b))
                    .expect("bench: mul must succeed")
            });
        });
    }

    group.finish();
}

// ─── Finite-difference method comparison ─────────────────────────────────────

/// Direct head-to-head comparison of central vs. forward difference at size 100.
fn bench_diff_methods(c: &mut Criterion) {
    let size = 100usize;
    let input: Vec<f32> = (0..size).map(|i| i as f32 * 0.1 + 0.1).collect();
    let analytical: Vec<f32> = input.iter().map(|&x| 2.0 * x).collect();

    let mut group = c.benchmark_group("finite_diff");

    group.bench_function("central_diff_100", |b| {
        let checker = GradientParityChecker::new(1e-5, 1e-4);
        b.iter(|| {
            checker
                .check_scalar_function(
                    |x| x.iter().map(|&v| v * v).sum::<f32>(),
                    black_box(&input),
                    black_box(&analytical),
                )
                .expect("bench: check must succeed")
        });
    });

    group.bench_function("forward_diff_100", |b| {
        let checker = GradientParityChecker::new(1e-5, 1e-4).with_forward_diff();
        b.iter(|| {
            checker
                .check_forward_diff(
                    |x| x.iter().map(|&v| v * v).sum::<f32>(),
                    black_box(&input),
                    black_box(&analytical),
                )
                .expect("bench: check must succeed")
        });
    });

    group.finish();
}

// ─── Registration ─────────────────────────────────────────────────────────────

criterion_group!(
    benches,
    bench_gradient_parity,
    bench_tensor_creation,
    bench_arithmetic,
    bench_diff_methods,
);
criterion_main!(benches);
