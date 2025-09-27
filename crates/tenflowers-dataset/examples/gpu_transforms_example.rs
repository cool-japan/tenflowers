//! Example demonstrating GPU-accelerated image transforms
//!
//! This example shows how to use GPU-accelerated image augmentation transforms
//! for faster data preprocessing on systems with GPU support.
//!
//! Run with: `cargo run --example gpu_transforms_example --features gpu`

#[cfg(feature = "gpu")]
use std::sync::Arc;
#[cfg(feature = "gpu")]
use tenflowers_core::Tensor;
#[cfg(feature = "gpu")]
use tenflowers_dataset::{
    Dataset, DatasetExt, GpuColorJitter, GpuContext, GpuRandomHorizontalFlip, GpuResize,
    TensorDataset, Transform,
};

#[cfg(feature = "gpu")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("GPU-accelerated image transforms example");

    // Create sample RGB image tensor (3x64x64)
    let image_size = 3 * 64 * 64;
    let mut image_data = Vec::with_capacity(image_size);

    // Generate a simple gradient pattern
    for c in 0..3 {
        for h in 0..64 {
            for w in 0..64 {
                let value = (c as f32 * 0.3 + h as f32 / 64.0 + w as f32 / 64.0) / 3.0;
                image_data.push(value);
            }
        }
    }

    let image_tensor = Tensor::from_vec(image_data, &[3, 64, 64])?;
    let label_tensor = Tensor::from_vec(vec![1.0f32], &[1])?;

    // Create a simple dataset
    let dataset = TensorDataset::new(image_tensor, label_tensor.clone());

    println!("Original dataset size: {}", dataset.len());

    // Try to initialize GPU context using blocking call
    match pollster::block_on(GpuContext::new()) {
        Ok(context) => {
            let context = Arc::new(context);
            println!("GPU context initialized successfully!");

            // Test GPU resize transform
            println!("\nTesting GPU resize transform...");
            let gpu_resize = GpuResize::new(128, 128, context.clone())?;
            let sample = dataset.get(0)?;
            let resized_sample = gpu_resize.apply(sample)?;
            println!("Resized image shape: {:?}", resized_sample.0.shape().dims());

            // Test GPU horizontal flip transform
            println!("\nTesting GPU horizontal flip transform...");
            let gpu_flip = GpuRandomHorizontalFlip::new(1.0, context.clone())?; // Always flip for demo
            let sample = dataset.get(0)?;
            let flipped_sample = gpu_flip.apply(sample)?;
            println!("Flipped image shape: {:?}", flipped_sample.0.shape().dims());

            // Test GPU color jitter transform
            println!("\nTesting GPU color jitter transform...");
            let gpu_color_jitter = GpuColorJitter::new(
                (0.9, 1.1),  // brightness_range
                (0.9, 1.1),  // contrast_range
                (0.9, 1.1),  // saturation_range
                (-0.1, 0.1), // hue_range
                context.clone(),
            )?;
            let sample = dataset.get(0)?;
            let jittered_sample = gpu_color_jitter.apply(sample)?;
            println!(
                "Color jittered image shape: {:?}",
                jittered_sample.0.shape().dims()
            );

            // Performance comparison (basic)
            println!("\nRunning basic performance test...");
            let start = std::time::Instant::now();

            // Apply multiple transforms
            for i in 0..10 {
                let gpu_resize = GpuResize::new(96, 96, context.clone())?;
                let gpu_flip = GpuRandomHorizontalFlip::new(0.5, context.clone())?;

                let sample = dataset.get(0)?;
                let resized = gpu_resize.apply(sample)?;
                let _final = gpu_flip.apply(resized)?;

                if i % 5 == 0 {
                    println!("Processed {} samples", i + 1);
                }
            }

            let duration = start.elapsed();
            println!("GPU transforms completed in: {:?}", duration);

            println!("\nGPU transforms example completed successfully!");
        }
        Err(e) => {
            println!("Could not initialize GPU context: {}", e);
            println!(
                "This is expected on systems without GPU support or with GPU feature disabled."
            );
            println!("To enable GPU support, run with --features gpu");
        }
    }

    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("GPU-accelerated image transforms example");
    println!("GPU support is not enabled. To use GPU transforms, run with:");
    println!("cargo run --example gpu_transforms_example --features gpu");
    Ok(())
}

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_context_fallback() {
        // Test that the example works even without GPU
        let result = pollster::block_on(GpuContext::new());

        // This should either succeed (if GPU is available) or fail gracefully
        match result {
            Ok(_) => println!("GPU context available"),
            Err(_) => println!("GPU context not available (expected on some systems)"),
        }
    }
}
