//! Custom Gradient Operations Example
//!
//! This example demonstrates how to:
//! 1. Use the new GradientPipeline for chaining gradient operations
//! 2. Apply gradient clipping, noise addition, and scaling
//! 3. Implement custom gradient processing workflows
//! 4. Validate gradient quality and statistics
//!
//! Run with: cargo run --example custom_gradient_operations

use tenflowers_autograd::gradient_ops::{
    add_gradient_noise, clip_by_global_norm, clip_by_value, compute_gradient_statistics,
    scale_gradients, GradientPipeline,
};
use tenflowers_core::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("==== Custom Gradient Operations Example ====\n");

    // Example 1: Basic gradient operations
    example_1_basic_operations()?;

    // Example 2: Gradient pipeline (chaining operations)
    example_2_gradient_pipeline()?;

    // Example 3: Gradient statistics and validation
    example_3_statistics_and_validation()?;

    // Example 4: Privacy-preserving gradients with noise
    example_4_privacy_preserving()?;

    // Example 5: Advanced pipeline configurations
    example_5_advanced_pipelines()?;

    println!("\n==== All Examples Completed Successfully ====");
    Ok(())
}

/// Example 1: Basic gradient operations
fn example_1_basic_operations() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Example 1: Basic Gradient Operations ===");

    // Create some mock gradients
    let grads = vec![
        Tensor::from_data(vec![10.0f32, 20.0, 30.0], &[3])?,
        Tensor::from_data(vec![40.0f32, 50.0], &[2])?,
    ];

    println!("Original gradients:");
    for (i, grad) in grads.iter().enumerate() {
        println!("  Gradient {}: {:?}", i, grad.as_slice().unwrap());
    }

    // Clip by global norm
    let (clipped, original_norm) = clip_by_global_norm(&grads, 10.0)?;
    println!("\nAfter clipping by global norm (max=10.0):");
    println!("  Original norm: {:.2}", original_norm);
    for (i, grad) in clipped.iter().enumerate() {
        println!("  Gradient {}: {:?}", i, grad.as_slice().unwrap());
    }

    // Clip by value
    let clipped_values = clip_by_value(&grads, 25.0)?;
    println!("\nAfter clipping by value (max=25.0):");
    for (i, grad) in clipped_values.iter().enumerate() {
        println!("  Gradient {}: {:?}", i, grad.as_slice().unwrap());
    }

    // Scale gradients
    let scaled = scale_gradients(&grads, 0.5)?;
    println!("\nAfter scaling by 0.5:");
    for (i, grad) in scaled.iter().enumerate() {
        println!("  Gradient {}: {:?}", i, grad.as_slice().unwrap());
    }

    println!();
    Ok(())
}

/// Example 2: Gradient pipeline (chaining operations)
fn example_2_gradient_pipeline() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Example 2: Gradient Pipeline (Chaining) ===");

    // Create gradients with large values
    let grads = vec![
        Tensor::from_data(vec![100.0f32, 200.0, 300.0], &[3])?,
        Tensor::from_data(vec![400.0f32, 500.0], &[2])?,
    ];

    println!("Original gradients:");
    for (i, grad) in grads.iter().enumerate() {
        println!("  Gradient {}: {:?}", i, grad.as_slice().unwrap());
    }

    // Create a gradient processing pipeline
    let pipeline = GradientPipeline::new()
        .clip_by_norm(50.0) // First, clip to norm 50
        .scale(0.1) // Then scale down
        .clip_by_value(2.0); // Finally, clip individual values

    println!("\nPipeline configuration:");
    println!("  1. Clip by global norm (max=50.0)");
    println!("  2. Scale by 0.1");
    println!("  3. Clip by value (max=2.0)");
    println!("  Total operations: {}", pipeline.len());

    // Apply the pipeline
    let processed = pipeline.apply(&grads)?;

    println!("\nProcessed gradients:");
    for (i, grad) in processed.iter().enumerate() {
        println!("  Gradient {}: {:?}", i, grad.as_slice().unwrap());
    }

    // Verify all values are within [-2, 2]
    for grad in &processed {
        for &val in grad.as_slice().unwrap() {
            assert!(
                val >= -2.0 && val <= 2.0,
                "Value {} is outside [-2, 2]",
                val
            );
        }
    }
    println!("✓ All values are within [-2.0, 2.0]");

    println!();
    Ok(())
}

/// Example 3: Gradient statistics and validation
fn example_3_statistics_and_validation() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Example 3: Gradient Statistics and Validation ===");

    // Create gradients with various characteristics
    let grads = vec![
        Tensor::from_data(vec![0.0f32, 0.1, 0.5, 1.0, 2.0], &[5])?,
        Tensor::from_data(vec![0.0f32, 0.0, 3.0], &[3])?,
    ];

    println!("Gradients:");
    for (i, grad) in grads.iter().enumerate() {
        println!("  Gradient {}: {:?}", i, grad.as_slice().unwrap());
    }

    // Compute statistics
    let stats = compute_gradient_statistics(&grads)?;

    println!("\nGradient Statistics:");
    println!("  Global norm: {:.4}", stats.global_norm);
    println!("  Max absolute value: {:.4}", stats.max_abs);
    println!("  Min absolute value: {:.4}", stats.min_abs);
    println!("  Mean absolute value: {:.4}", stats.mean_abs);
    println!("  Number of NaN values: {}", stats.num_nan);
    println!("  Number of Inf values: {}", stats.num_inf);
    println!("  Number of zero values: {}", stats.num_zero);
    println!("  Total elements: {}", stats.total_elements);
    println!("  Sparsity: {:.2}%", stats.sparsity() * 100.0);
    println!("  Healthy: {}", stats.is_healthy());

    // Demonstrate unhealthy gradients
    let unhealthy_grads = vec![
        Tensor::from_data(vec![1.0f32, f32::NAN, 3.0], &[3])?,
        Tensor::from_data(vec![f32::INFINITY, 5.0], &[2])?,
    ];

    let unhealthy_stats = compute_gradient_statistics(&unhealthy_grads)?;
    println!("\nUnhealthy Gradient Statistics:");
    println!("  NaN count: {}", unhealthy_stats.num_nan);
    println!("  Inf count: {}", unhealthy_stats.num_inf);
    println!("  Healthy: {}", unhealthy_stats.is_healthy());

    println!();
    Ok(())
}

/// Example 4: Privacy-preserving gradients with noise
fn example_4_privacy_preserving() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Example 4: Privacy-Preserving Gradients ===");

    // Simulate gradients from a model
    let grads = vec![
        Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], &[5])?,
        Tensor::from_data(vec![6.0f32, 7.0, 8.0], &[3])?,
    ];

    println!("Original gradients:");
    for (i, grad) in grads.iter().enumerate() {
        println!("  Gradient {}: {:?}", i, grad.as_slice().unwrap());
    }

    // Add Gaussian noise for differential privacy
    let noise_stddev = 0.5;
    let noisy_grads = add_gradient_noise(&grads, noise_stddev, Some(42))?;

    println!("\nAfter adding noise (stddev={}):", noise_stddev);
    for (i, grad) in noisy_grads.iter().enumerate() {
        println!("  Gradient {}: {:?}", i, grad.as_slice().unwrap());
    }

    // Verify noise was added (values should be different)
    let original_slice = grads[0].as_slice().unwrap();
    let noisy_slice = noisy_grads[0].as_slice().unwrap();
    let all_different = original_slice
        .iter()
        .zip(noisy_slice.iter())
        .all(|(a, b)| (a - b).abs() > 1e-6);

    assert!(all_different, "Noise should have been added");
    println!("✓ Noise successfully added to gradients");

    // Demonstrate reproducibility with seed
    let noisy_grads_2 = add_gradient_noise(&grads, noise_stddev, Some(42))?;
    let noisy_slice_2 = noisy_grads_2[0].as_slice().unwrap();

    let all_same = noisy_slice
        .iter()
        .zip(noisy_slice_2.iter())
        .all(|(a, b)| (a - b).abs() < 1e-6);

    assert!(all_same, "Same seed should produce same noise");
    println!("✓ Same seed produces reproducible noise");

    println!();
    Ok(())
}

/// Example 5: Advanced pipeline configurations
fn example_5_advanced_pipelines() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Example 5: Advanced Pipeline Configurations ===");

    let grads = vec![Tensor::from_data(vec![50.0f32, 100.0, 150.0, 200.0], &[4])?];

    println!("Original gradients: {:?}", grads[0].as_slice().unwrap());

    // Configuration 1: Conservative (for stable training)
    println!("\n--- Configuration 1: Conservative (Stable Training) ---");
    let conservative_pipeline = GradientPipeline::new().clip_by_norm(5.0).clip_by_value(1.0);

    let conservative_result = conservative_pipeline.apply(&grads)?;
    println!("Result: {:?}", conservative_result[0].as_slice().unwrap());

    // Configuration 2: Privacy-preserving
    println!("\n--- Configuration 2: Privacy-Preserving ---");
    let privacy_pipeline = GradientPipeline::new()
        .clip_by_norm(10.0)
        .add_noise(1.0, Some(123))
        .clip_by_value(5.0);

    let privacy_result = privacy_pipeline.apply(&grads)?;
    println!("Result: {:?}", privacy_result[0].as_slice().unwrap());

    // Configuration 3: Aggressive optimization
    println!("\n--- Configuration 3: Aggressive Optimization ---");
    let aggressive_pipeline = GradientPipeline::new()
        .scale(2.0) // Amplify gradients
        .clip_by_norm(50.0); // Prevent explosion

    let aggressive_result = aggressive_pipeline.apply(&grads)?;
    println!("Result: {:?}", aggressive_result[0].as_slice().unwrap());

    // Configuration 4: Multi-stage processing
    println!("\n--- Configuration 4: Multi-Stage Processing ---");
    let multistage_pipeline = GradientPipeline::new()
        .clip_by_norm(30.0) // Stage 1: Coarse clipping
        .add_noise(0.5, Some(456)) // Stage 2: Add noise
        .clip_by_value(10.0) // Stage 3: Fine-grained clipping
        .scale(0.1); // Stage 4: Learning rate scaling

    let multistage_result = multistage_pipeline.apply(&grads)?;
    println!("Result: {:?}", multistage_result[0].as_slice().unwrap());

    // Demonstrate pipeline reusability
    println!("\n--- Pipeline Reusability ---");
    let grads2 = vec![Tensor::from_data(vec![10.0f32, 20.0, 30.0], &[3])?];

    let result2 = conservative_pipeline.apply(&grads2)?;
    println!(
        "Conservative pipeline on new gradients: {:?}",
        result2[0].as_slice().unwrap()
    );

    println!();
    Ok(())
}
