//! Example demonstrating ImageNet dataset loading and usage
//!
//! This example shows how to load ImageNet validation set with automatic downloading
//! and preprocessing, and how to use it with the TenfloweRS dataset pipeline.
//!
//! Run with: `cargo run --example imagenet_example --features download,images`

#[cfg(feature = "gpu")]
use pollster;
use std::path::Path;
use tenflowers_core::Tensor;
use tenflowers_dataset::{
    Dataset, DatasetExt, GpuContext, ImageNetConfig, Normalize, RealImageNetBuilder,
    RealImageNetDataset, Transform,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("ImageNet dataset example");

    // Example 1: Load a small subset of ImageNet validation set
    println!("\n=== Example 1: Loading ImageNet validation set ===");

    let imagenet_dataset = RealImageNetBuilder::new()
        .root("./data")
        .train(false) // Use validation set (training set is too large)
        .download(false) // Set to true to download automatically
        .image_size(224, 224) // Standard ImageNet size
        .max_samples(Some(100)) // Load only 100 samples for demo
        .build::<f32>();

    match imagenet_dataset {
        Ok(dataset) => {
            println!("ImageNet dataset loaded successfully!");
            println!("Number of samples: {}", dataset.len());
            println!("Number of classes: {}", dataset.num_classes());

            // Get a sample
            if let Ok((image, label)) = dataset.get(0) {
                println!("First sample:");
                println!("  Image shape: {:?}", image.shape().dims());
                println!("  Label: {:?}", label.as_slice());
            }

            // Show class names
            let class_names = dataset.class_names();
            println!("First 10 class names: {:?}", &class_names[0..10]);

            // Example 2: Apply transforms to ImageNet data
            println!("\n=== Example 2: Applying transforms ===");

            // Create a normalization transform (ImageNet already normalized during loading)
            let normalize = Normalize::new(
                vec![0.0, 0.0, 0.0], // Already normalized, so center at 0
                vec![1.0, 1.0, 1.0], // Already normalized, so scale by 1
            )?;

            // Apply transform to first sample
            if let Ok(sample) = dataset.get(0) {
                let transformed = normalize.apply(sample)?;
                println!(
                    "Transformed sample shape: {:?}",
                    transformed.0.shape().dims()
                );
            }

            // Example 3: Batch processing
            println!("\n=== Example 3: Batch processing ===");

            let batch_size = 8;
            let mut batch_images = Vec::new();
            let mut batch_labels = Vec::new();

            for i in 0..batch_size.min(dataset.len()) {
                if let Ok((image, label)) = dataset.get(i) {
                    batch_images.push(image);
                    batch_labels.push(label);
                }
            }

            println!("Processed batch of {} samples", batch_images.len());

            // Example 4: Integration with GPU transforms (if available)
            println!("\n=== Example 4: GPU transforms (if available) ===");

            #[cfg(feature = "gpu")]
            {
                match pollster::block_on(GpuContext::new()) {
                    Ok(context) => {
                        println!("GPU context available - could apply GPU transforms");
                        // GPU transforms would be applied here
                    }
                    Err(_) => {
                        println!("GPU context not available - using CPU transforms");
                    }
                }
            }

            #[cfg(not(feature = "gpu"))]
            {
                println!("GPU feature not enabled - using CPU transforms");
            }
        }
        Err(e) => {
            println!("Failed to load ImageNet dataset: {}", e);
            println!("This is expected if ImageNet data is not available.");
            println!("To use ImageNet:");
            println!("1. Set download=true in the builder (for validation set only)");
            println!("2. Or manually download ImageNet data to ./data/ImageNet/");
            println!("3. Run with --features download,images");
        }
    }

    // Example 5: Show how to create a custom ImageNet config
    println!("\n=== Example 5: Custom ImageNet configuration ===");

    let custom_config = ImageNetConfig {
        root: Path::new("./custom_data").to_path_buf(),
        train: false,
        download: false,
        image_size: (256, 256), // Different size
        max_samples: Some(50),
    };

    println!("Custom config created:");
    println!("  Root: {:?}", custom_config.root);
    println!("  Training set: {}", custom_config.train);
    println!("  Image size: {:?}", custom_config.image_size);
    println!("  Max samples: {:?}", custom_config.max_samples);

    // Example 6: Show integration with other datasets
    println!("\n=== Example 6: Integration patterns ===");

    println!("ImageNet can be used with:");
    println!("- DataLoader for batch processing");
    println!("- Transform pipelines for augmentation");
    println!("- GPU acceleration for preprocessing");
    println!("- Mixed with other datasets using ConcatDataset");
    println!("- Cached for faster subsequent access");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_imagenet_builder_configuration() {
        let temp_dir = TempDir::new().unwrap();

        let builder = RealImageNetBuilder::new()
            .root(temp_dir.path())
            .train(false)
            .download(false)
            .image_size(256)
            .center_crop(false)
            .max_samples(Some(10));

        // Test that builder is configured correctly
        assert_eq!(builder.config.image_size, 256);
        assert!(!builder.config.center_crop);
        assert_eq!(builder.config.max_samples, Some(10));
    }

    #[test]
    fn test_imagenet_config_validation() {
        let temp_dir = TempDir::new().unwrap();

        let config = ImageNetConfig {
            root: temp_dir.path().to_path_buf(),
            train: true, // This should be rejected
            download: false,
            image_size: 224,
            center_crop: true,
            max_samples: None,
        };

        // Building with train=true should fail
        let result = RealImageNetDataset::<f32>::new(config);
        assert!(result.is_err());
    }
}
