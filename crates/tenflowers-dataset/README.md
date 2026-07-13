# TenfloweRS Dataset

Data loading and preprocessing utilities for TenfloweRS, providing efficient dataset management, transformations, and data pipelines for machine learning workflows.

> Stable (v0.2.0 -- 2026-07-13) | 698 tests passing (`--all-features`) | 0 clippy warnings

## Overview

`tenflowers-dataset` implements:
- **Dataset Abstractions**: Flexible trait-based dataset interface
- **Data Transformations**: Preprocessing and augmentation pipelines
- **Batch Processing**: Efficient batching with automatic tensor stacking
- **Data Loading**: Support for CSV, Parquet, HDF5, TFRecord, Zarr, WebDataset (TAR-shard streaming), images, and audio formats
- **Parallel Processing**: Multi-threaded data loading and preprocessing
- **Memory Efficiency**: Lazy loading, memory mapping, and caching strategies
- **Distributed Streaming**: Sharded streaming for large-scale training
- **Multimodal Support**: Unified transforms for multi-modal data
- **Synthetic Data Generation**: Built-in generators for images, text, time series, and modern ML datasets
- **Reproducibility**: Deterministic data pipelines with seed control

## Features

- **Flexible Dataset Trait**: Define custom datasets for any data source
- **Composable Transforms**: Chain preprocessing operations with a pipeline API
- **Automatic Batching**: Convert individual samples to batched tensors
- **Data Augmentation**: Augmentation techniques for images, text, and audio, including real Box-Muller Gaussian noise injection
- **Prefetching**: Overlap data loading with model computation
- **Distributed Support**: Sharding for multi-GPU training
- **Caching with Telemetry**: LRU caching with hit-rate monitoring
- **Vision Transforms**: Resize, crop, flip, color jitter, normalization, plus GPU-accelerated affine, perspective, elastic distortion, and histogram equalization
- **SIMD Preprocessing**: Auto-vectorized functional transforms (normalize, standardize, CHW/HWC conversion, grayscale, mean/variance)
- **Transform Arena**: Reusable-buffer arena for allocation-free transform pipelines
- **Zarr Blosc Decoding**: Pure-Rust Blosc-chunk decoder (BloscLZ, LZ4, Snappy, Zlib, Zstd inner codecs with byte/bit-shuffle filters)
- **TFRecord SequenceExample**: Reader for sequence-shaped TFRecord data with masked-CRC integrity checks

## Usage

### Basic Tensor Dataset

```rust
use tenflowers_dataset::{Dataset, TensorDataset};
use tenflowers_core::Tensor;

// Create dataset from tensors
let features = Tensor::from_vec(
    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    &[3, 2]  // 3 samples, 2 features each
)?;
let labels = Tensor::from_vec(vec![0.0, 1.0, 2.0], &[3])?;

let dataset = TensorDataset::new(features, labels);

// Access individual samples
let (sample_features, sample_label) = dataset.get(0)?;

// Create batched iterator
for batch in dataset.batch(2) {
    // batch contains Vec<(Tensor, Tensor)> with batch_size samples
    for (features, label) in batch {
        // Process batch
    }
}
```

### Data Transformations

```rust
use tenflowers_dataset::{Transform, Normalize, MinMaxScale, AddNoise, DatasetExt};
use tenflowers_dataset::transforms::pipeline::Compose;

// Normalize features with per-feature mean and std
let normalize = Normalize::new(vec![0.5, 0.5], vec![0.5, 0.5])?;

// Chain multiple transforms; `.add()` boxes each stage internally, so one
// Compose chain can mix different Transform types
let transform = Compose::new()
    .add(normalize)
    .add(MinMaxScale::new(vec![0.0, 0.0], vec![1.0, 1.0], (0.0, 1.0))?)
    .add(AddNoise::new(0.1));

// Apply to dataset
let transformed = dataset.transform(transform);
```

### Custom Dataset Implementation

```rust
use tenflowers_dataset::Dataset;
use tenflowers_core::{Result, Tensor};
use std::path::PathBuf;

struct ImageDataset {
    image_paths: Vec<PathBuf>,
    labels: Vec<u32>,
}

impl Dataset<f32> for ImageDataset {
    fn len(&self) -> usize {
        self.image_paths.len()
    }

    fn get(&self, index: usize) -> Result<(Tensor<f32>, Tensor<f32>)> {
        // Load image from disk (your own decoding logic, or `formats::image`)
        let image = load_image(&self.image_paths[index])?;
        let image_tensor = image_to_tensor(image)?;

        // Convert label to a 0-D tensor
        let label_tensor = Tensor::from_scalar(self.labels[index] as f32);

        Ok((image_tensor, label_tensor))
    }
}
```

### Data Augmentation Pipeline

```rust
use tenflowers_dataset::{DatasetExt, Normalize};
use tenflowers_dataset::transforms::pipeline::Compose;
use tenflowers_dataset::transforms::vision::{ColorJitter, RandomCropWithPadding, RandomHorizontalFlip};

// Create an augmentation pipeline for images from real vision transforms
let augmentation = Compose::new()
    .add(RandomCropWithPadding::without_padding(224, 224))
    .add(RandomHorizontalFlip::new(0.5))
    .add(ColorJitter::new().with_brightness(0.8, 1.2).with_contrast(0.8, 1.2))
    .add(Normalize::new(imagenet_mean, imagenet_std)?);

// Apply to dataset during training
let train_dataset = dataset.transform(augmentation);
```

### Parallel Data Loading

```rust
use tenflowers_dataset::{DataLoaderBuilder, RandomSampler};

// Create parallel data loader
let loader = DataLoaderBuilder::new(dataset)
    .batch_size(32)
    .num_workers(4)      // Parallel loading threads
    .prefetch_factor(2)  // Prefetch 2 batches
    .drop_last(true)     // Drop incomplete final batch
    .build(RandomSampler::new()); // shuffled sampling order

// Iterate with automatic prefetching
for batch in loader.iter() {
    let batch = batch?; // BatchResult::{Samples(_) | Collated(features, labels)}
    // Batched tensors ready for training
}
```

## Architecture

### Core Components

- **Dataset Trait**: Unified interface for all data sources
- **Transform Trait**: Composable data transformations
- **DataLoader**: Efficient batching and prefetching
- **Samplers**: Control iteration order and distribution

### Supported Data Formats

- **In-Memory**: Tensor datasets, array datasets
- **Files**: Images (PNG, JPEG), CSV, JSON, Parquet
- **Binary**: TFRecord (including `SequenceExample`)
- **Scientific Arrays**: Zarr, with a pure-Rust Blosc chunk decoder (BloscLZ, LZ4, Snappy, Zlib, Zstd); HDF5 behind the optional `hdf5` feature (native C library)
- **Streaming Archives**: WebDataset (TAR-shard) for large-scale distributed training
- **Text**: Plain text, tokenized sequences
- **Audio**: WAV, MP3, FLAC with on-the-fly Symphonia-based decoding and probing

### Performance Features

- **Memory Mapping**: For large datasets that do not fit in RAM
- **Caching**: LRU cache with telemetry for frequently accessed samples
- **Prefetching**: Overlap I/O with computation
- **Parallel Loading**: Multi-threaded data loading
- **NUMA-aware**: Optional NUMA memory placement
- **Transform Arena**: Reusable buffer arena for allocation-free transform chains

## Feature Flags

- `parallel`: Parallel data loading via Rayon
- `serialize`: Serialization support for datasets and transforms
- `images`: Image loading and processing
- `mmap`: Memory-mapped dataset support
- `parquet`: Parquet file format support
- `csv_format`: CSV file format support
- `regex`: Regex-based field parsing for text and structured formats
- `tfrecord`: TFRecord file format support
- `msgpack` [Planned]: MessagePack serialization (implies `serialize`) — the feature flag is default-on and pulls in the `rmp-serde` dependency, but no serializer/deserializer is implemented anywhere in the crate; enabling it currently has no functional effect
- `hdf5`: HDF5 format support (requires the native HDF5 C library; not enabled by default — Zarr/Parquet are the pure-Rust alternatives)
- `audio`: Audio file loading and processing
- `download`: Dataset download utilities
- `webdataset`: WebDataset TAR-shard streaming format
- `gpu`: GPU direct data loading
- `numa`: NUMA-aware memory placement
- `distributed`: Multi-worker distributed streaming and coordination (Tokio-based)
- `cloud`: Cloud object-storage-backed dataset loading
- `compression`: Standalone OxiARC compression codecs (LZ4, Deflate, Snappy)

## Integration with TenfloweRS

- **Seamless Tensor Creation**: Direct conversion to tenflowers-core tensors
- **Device Placement**: Automatic CPU/GPU placement
- **Gradient Tape**: Compatible with autograd for data-dependent gradients
- **Model Training**: Direct integration with neural network training loops

## License

Licensed under Apache-2.0
