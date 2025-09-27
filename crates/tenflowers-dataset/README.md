# TenfloweRS Dataset

Data loading and preprocessing utilities for TenfloweRS, providing efficient dataset management, transformations, and data pipelines for machine learning workflows.

> Alpha Notice (0.1.0-alpha.1 · 2025-09-27)
> Core dataset abstractions and transform pipeline are present; advanced distributed sharding, streaming, and some format loaders are placeholders or partial.

## Overview

`tenflowers-dataset` implements:
- **Dataset Abstractions**: Flexible trait-based dataset interface
- **Data Transformations**: Preprocessing and augmentation pipelines
- **Batch Processing**: Efficient batching with automatic tensor stacking
- **Data Loading**: Support for various data formats and sources
- **Parallel Processing**: Multi-threaded data loading and preprocessing
- **Memory Efficiency**: Lazy loading and caching strategies

## Features

- **Flexible Dataset Trait**: Define custom datasets for any data source
- **Composable Transforms**: Chain preprocessing operations
- **Automatic Batching**: Convert individual samples to batched tensors
- **Data Augmentation**: Common augmentation techniques for images, text, audio
- **Prefetching**: Overlap data loading with model computation
- **Distributed Support**: Sharding for multi-GPU training

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
use tenflowers_dataset::{Transform, Normalize, MinMaxScale, Compose};

// Normalize features with mean and std
let normalize = Normalize::new(vec![0.5, 0.5], vec![0.5, 0.5]);

// Chain multiple transforms
let transform = Compose::new(vec![
    Box::new(Normalize::new(mean, std)),
    Box::new(MinMaxScale::new(0.0, 1.0)),
    Box::new(AddNoise::gaussian(0.0, 0.1)),
]);

// Apply to dataset
let transformed = dataset.transform(transform);
```

### Custom Dataset Implementation

```rust
use tenflowers_dataset::Dataset;
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
        // Load image from disk
        let image = load_image(&self.image_paths[index])?;
        let image_tensor = image_to_tensor(image)?;
        
        // Convert label to tensor
        let label_tensor = Tensor::scalar(self.labels[index] as f32, Device::Cpu)?;
        
        Ok((image_tensor, label_tensor))
    }
}
```

### Data Augmentation Pipeline

```rust
use tenflowers_dataset::{ImageAugmentation, RandomCrop, RandomFlip};

// Create augmentation pipeline for images
let augmentation = ImageAugmentation::builder()
    .random_crop(224, 224)
    .random_horizontal_flip(0.5)
    .random_rotation(-15.0, 15.0)
    .color_jitter(0.2, 0.2, 0.2, 0.1)
    .normalize(imagenet_mean, imagenet_std)
    .build()?;

// Apply to dataset during training
let train_dataset = dataset.transform(augmentation);
```

### CSV Dataset

```rust
use tenflowers_dataset::CsvDataset;

// Load dataset from CSV file
let dataset = CsvDataset::builder()
    .file_path("data.csv")
    .has_header(true)
    .feature_columns(vec!["feature1", "feature2", "feature3"])
    .label_column("target")
    .delimiter(',')
    .build()?;

// Automatic type inference and conversion to tensors
for batch in dataset.batch(32) {
    // Process batches
}
```

### TFRecord Dataset

```rust
use tenflowers_dataset::TFRecordDataset;

// Load TensorFlow TFRecord files
let dataset = TFRecordDataset::new(vec!["data.tfrecord"])?;

// Parse examples with feature descriptions
let parsed = dataset.parse_example(|example| {
    let image = example.get_bytes("image")?;
    let label = example.get_int64("label")?;
    
    // Decode and preprocess
    let image_tensor = decode_image(image)?;
    let label_tensor = Tensor::scalar(label as f32, Device::Cpu)?;
    
    Ok((image_tensor, label_tensor))
});
```

### Parallel Data Loading

```rust
use tenflowers_dataset::{DataLoader, PrefetchConfig};

// Create parallel data loader
let loader = DataLoader::builder()
    .dataset(dataset)
    .batch_size(32)
    .num_workers(4)  // Parallel loading threads
    .prefetch(2)     // Prefetch 2 batches
    .shuffle(true)
    .drop_last(true) // Drop incomplete final batch
    .build()?;

// Iterate with automatic prefetching
for batch in loader {
    let (features, labels) = batch?;
    // Batched tensors ready for training
}
```

### Distributed Data Loading

```rust
use tenflowers_dataset::{DistributedSampler};

// Create distributed sampler for multi-GPU training
let sampler = DistributedSampler::new(
    dataset_len,
    num_replicas: 4,
    rank: 0,
    shuffle: true,
);

// Each GPU gets a unique subset of data
let distributed_dataset = dataset.with_sampler(sampler);
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
- **Binary**: TFRecord, MessagePack, Protobuf
- **Text**: Plain text, tokenized sequences
- **Audio**: WAV, MP3, FLAC with on-the-fly processing

### Performance Features

- **Memory Mapping**: For large datasets that don't fit in RAM
- **Caching**: LRU cache for frequently accessed samples
- **Prefetching**: Overlap I/O with computation
- **Parallel Loading**: Multi-threaded data loading
- **GPU Direct**: Direct loading to GPU memory

## Common Patterns

### Train/Validation/Test Split

```rust
use tenflowers_dataset::{train_test_split, DatasetSplit};

// Split dataset into train/val/test
let (train, val, test) = dataset.split(&[0.7, 0.15, 0.15])?;

// Or use predefined splits
let splits = DatasetSplit::from_indices(
    train_indices,
    val_indices,
    test_indices,
);
```

### Data Pipeline for Training

```rust
// Complete training pipeline
let train_pipeline = dataset
    .shuffle(10000)
    .transform(augmentation)
    .batch(32)
    .prefetch(2);

let val_pipeline = val_dataset
    .batch(32)
    .prefetch(1);

// Use in training loop
for epoch in 0..num_epochs {
    for batch in &train_pipeline {
        // Training step
    }
    
    for batch in &val_pipeline {
        // Validation step
    }
}
```

### Infinite Dataset Iterator

```rust
use tenflowers_dataset::InfiniteDataset;

// Create infinite iterator for continuous training
let infinite = InfiniteDataset::new(dataset)
    .shuffle_each_epoch(true);

// Take specific number of steps
for batch in infinite.take(1000).batch(32) {
    // Process batch
}
```

## Integration with TenfloweRS

- **Seamless Tensor Creation**: Direct conversion to tenflowers-core tensors
- **Device Placement**: Automatic CPU/GPU placement
- **Gradient Tape**: Compatible with autograd for data-dependent gradients
- **Model Training**: Direct integration with neural network training loops

## Performance Considerations

- Use appropriate batch sizes for your hardware
- Enable prefetching for I/O-bound datasets
- Use memory mapping for datasets larger than RAM
- Consider data format for optimal loading speed
- Profile data loading to identify bottlenecks

### Current Alpha Limitations
- Some listed file/format loaders (Parquet advanced predicates, TFRecord sequence features) are stubs
- Streaming + resumable iteration incomplete
- Distributed sampler lacks fault tolerance & elasticity
- GPU direct data path experimental

### Near-Term Priorities
1. Unified error taxonomy for I/O & decoding
2. Async streaming reader for large sequential datasets
3. Pluggable shuffle buffer backends (disk / memory / mmap)
4. Format coverage: Arrow / simple HDF5 reader
5. Deterministic epoch sharding guarantees

## Future Enhancements

See TODO.md for detailed roadmap including:
- More data formats (HDF5, Zarr, Arrow)
- Advanced augmentation techniques
- Federated learning support
- Streaming datasets
- Active learning samplers

## Contributing

We welcome contributions! Priority areas:
- Implementing new data formats
- Adding augmentation techniques
- Optimizing data loading performance
- Creating dataset utilities
- Writing examples and tutorials

## License

Dual-licensed under MIT OR Apache-2.0