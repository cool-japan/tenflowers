// Criterion benchmark suite for tenflowers-dataset throughput.
//
// Measures:
//   (a) raw `Dataset::get` loop over a `SyntheticDataset`
//   (b) `DataLoader` with 0, 1, and 4 worker threads
//   (c) transform chain cost in isolation

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use tenflowers_core::Tensor;
use tenflowers_dataset::{
    dataloader::{DataLoader, DataLoaderConfig, SequentialSampler},
    transforms::Transform,
    Dataset, SyntheticDataset,
};

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Build a deterministic `SyntheticDataset<f32>` with `n_samples × feature_dim` features
/// and a 1-D label vector.
fn make_dataset(n_samples: usize, feature_dim: usize) -> SyntheticDataset<f32> {
    let total = n_samples * feature_dim;
    let features: Vec<f32> = (0..total).map(|i| (i as f32) * 0.001).collect();
    let labels: Vec<f32> = (0..n_samples).map(|i| (i % 2) as f32).collect();

    let feat_tensor = Tensor::<f32>::from_vec(features, &[n_samples, feature_dim])
        .expect("bench: failed to create feature tensor");
    let label_tensor = Tensor::<f32>::from_vec(labels, &[n_samples])
        .expect("bench: failed to create label tensor");

    SyntheticDataset::new(feat_tensor, label_tensor)
}

/// A no-op transform so we can isolate framework overhead.
struct IdentityTransform;

impl Transform<f32> for IdentityTransform {
    fn apply(
        &self,
        sample: (Tensor<f32>, Tensor<f32>),
    ) -> tenflowers_core::Result<(Tensor<f32>, Tensor<f32>)> {
        Ok(sample)
    }
}

/// A simple per-element scale transform to add measurable work to the pipeline.
struct ScaleTransform {
    factor: f32,
}

impl Transform<f32> for ScaleTransform {
    fn apply(
        &self,
        (features, labels): (Tensor<f32>, Tensor<f32>),
    ) -> tenflowers_core::Result<(Tensor<f32>, Tensor<f32>)> {
        let shape = features.shape().dims().to_vec();
        let scaled: Vec<f32> = features
            .to_vec()
            .expect("bench: failed to get features vec")
            .iter()
            .map(|&v| v * self.factor)
            .collect();
        let out =
            Tensor::<f32>::from_vec(scaled, &shape).expect("bench: failed to create scaled tensor");
        Ok((out, labels))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// (a) Raw `Dataset::get` loop
// ─────────────────────────────────────────────────────────────────────────────

fn bench_raw_get(c: &mut Criterion) {
    let mut group = c.benchmark_group("raw_get");

    for &n_samples in &[256usize, 1024] {
        let dataset = make_dataset(n_samples, 64);
        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("n_samples", n_samples),
            &n_samples,
            |b, &n| {
                b.iter(|| {
                    for i in 0..n {
                        let _ = black_box(dataset.get(i).expect("bench: get should succeed"));
                    }
                });
            },
        );
    }

    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// (b) DataLoader with varying worker counts
// ─────────────────────────────────────────────────────────────────────────────

fn bench_dataloader_workers(c: &mut Criterion) {
    let n_samples = 256_usize;
    let feature_dim = 64_usize;
    let batch_size = 32_usize;

    let mut group = c.benchmark_group("dataloader_workers");
    group.throughput(Throughput::Elements(n_samples as u64));

    for &n_workers in &[0usize, 1, 4] {
        group.bench_with_input(
            BenchmarkId::new("num_workers", n_workers),
            &n_workers,
            |b, &workers| {
                b.iter(|| {
                    let dataset = make_dataset(n_samples, feature_dim);
                    let config = DataLoaderConfig {
                        batch_size,
                        num_workers: workers,
                        prefetch_factor: 2,
                        collate_batches: false,
                        ..DataLoaderConfig::default()
                    };
                    let sampler = SequentialSampler::new();
                    let loader = DataLoader::new(dataset, sampler, config);

                    let mut count = 0usize;
                    for items in loader.iter().flatten() {
                        if let Ok(samples) = items.into_samples() {
                            for item in samples {
                                let _ = black_box(item);
                                count += 1;
                            }
                        }
                    }
                    black_box(count);
                });
            },
        );
    }

    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// (c) Transform chain cost isolation
// ─────────────────────────────────────────────────────────────────────────────

fn bench_transform_chain(c: &mut Criterion) {
    let n_samples = 256_usize;
    let feature_dim = 64_usize;

    let mut group = c.benchmark_group("transform_chain");
    group.throughput(Throughput::Elements(n_samples as u64));

    // Identity-only: baseline overhead
    group.bench_function("identity_transform", |b| {
        let dataset = make_dataset(n_samples, feature_dim);
        let t = IdentityTransform;
        b.iter(|| {
            for i in 0..n_samples {
                let sample = dataset.get(i).expect("bench: get should succeed");
                let _ = black_box(t.apply(sample).expect("bench: transform should succeed"));
            }
        });
    });

    // Scale transform: actual per-element work
    group.bench_function("scale_transform", |b| {
        let dataset = make_dataset(n_samples, feature_dim);
        let t = ScaleTransform { factor: 2.0 };
        b.iter(|| {
            for i in 0..n_samples {
                let sample = dataset.get(i).expect("bench: get should succeed");
                let _ = black_box(t.apply(sample).expect("bench: transform should succeed"));
            }
        });
    });

    // Chain: identity → scale (two-step pipeline overhead)
    group.bench_function("identity_then_scale", |b| {
        let dataset = make_dataset(n_samples, feature_dim);
        let t1 = IdentityTransform;
        let t2 = ScaleTransform { factor: 0.5 };
        b.iter(|| {
            for i in 0..n_samples {
                let sample = dataset.get(i).expect("bench: get should succeed");
                let after_t1 = t1.apply(sample).expect("bench: t1 should succeed");
                let _ = black_box(t2.apply(after_t1).expect("bench: t2 should succeed"));
            }
        });
    });

    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Criterion entry points
// ─────────────────────────────────────────────────────────────────────────────

criterion_group!(
    benches,
    bench_raw_get,
    bench_dataloader_workers,
    bench_transform_chain,
);
criterion_main!(benches);
