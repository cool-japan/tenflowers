//! Advanced Gradient Clipping Demonstration
//!
//! This example showcases the advanced gradient clipping capabilities of TenfloweRS,
//! demonstrating both basic gradient clipping and adaptive scaling techniques that
//! are essential for stable training of large neural networks.

use scirs2_autograd::ndarray::Array1;
use tenflowers_core::{GradientClipper, GradientClippingConfig, NormType, Result, Tensor};

fn main() -> Result<()> {
    println!("🎯 TenfloweRS Advanced Gradient Clipping Demo");
    println!("=============================================");

    // Demo 1: Basic Gradient Clipping
    println!("\n📊 Demo 1: Basic Gradient Clipping");
    println!("----------------------------------");
    basic_gradient_clipping_demo()?;

    // Demo 2: Adaptive Gradient Clipping
    println!("\n🤖 Demo 2: Adaptive Gradient Clipping");
    println!("------------------------------------");
    adaptive_gradient_clipping_demo()?;

    // Demo 3: Parameter Group Clipping
    println!("\n🔧 Demo 3: Parameter Group Clipping");
    println!("----------------------------------");
    parameter_group_clipping_demo()?;

    // Demo 4: Different Norm Types
    println!("\n📐 Demo 4: Different Norm Types Comparison");
    println!("------------------------------------------");
    norm_types_comparison_demo()?;

    // Demo 5: Training Simulation
    println!("\n🏋️ Demo 5: Training Loop Simulation");
    println!("----------------------------------");
    training_simulation_demo()?;

    println!("\n✅ All gradient clipping demos completed successfully!");

    Ok(())
}

/// Demonstrate basic gradient clipping functionality
fn basic_gradient_clipping_demo() -> Result<()> {
    // Create a standard gradient clipper
    let mut clipper = GradientClipper::<f32>::default_stable();

    // Create some test gradients with varying magnitudes
    let gradients = vec![
        Tensor::from_array(Array1::from_vec(vec![0.5, 0.3, 0.2]).into_dyn()),
        Tensor::from_array(Array1::from_vec(vec![2.0, 3.0, 1.5]).into_dyn()), // Large gradients
        Tensor::from_array(Array1::from_vec(vec![0.1, 0.1, 0.1]).into_dyn()),
    ];

    println!("Original gradients:");
    for (i, grad) in gradients.iter().enumerate() {
        if let tenflowers_core::tensor::TensorStorage::Cpu(array) = &grad.storage {
            println!("  Gradient {}: {:?}", i, array.as_slice().unwrap());
        }
    }

    let mut clipped_gradients = gradients.clone();
    let global_norm = clipper.clip_gradients(&mut clipped_gradients)?;

    println!("\nAfter clipping (max_norm = 1.0):");
    println!("  Global norm before clipping: {:.4}", global_norm);
    println!("  Was clipped: {}", global_norm > 1.0);
    for (i, grad) in clipped_gradients.iter().enumerate() {
        if let tenflowers_core::tensor::TensorStorage::Cpu(array) = &grad.storage {
            println!("  Gradient {}: {:?}", i, array.as_slice().unwrap());
        }
    }

    let stats = clipper.get_statistics();
    println!(
        "  Clipping statistics: {} clips out of {} updates",
        stats.clip_count, stats.total_updates
    );

    Ok(())
}

/// Demonstrate adaptive gradient clipping
fn adaptive_gradient_clipping_demo() -> Result<()> {
    let mut clipper = GradientClipper::<f32>::default_adaptive();

    println!("Simulating training with gradually increasing gradient magnitudes:");

    for epoch in 1..=10 {
        // Simulate gradients that get progressively larger (unstable training)
        let scale = 0.5 + (epoch as f32) * 0.3;
        let gradients = vec![
            Tensor::from_array(Array1::from_vec(vec![scale, scale * 0.8, scale * 0.6]).into_dyn()),
            Tensor::from_array(
                Array1::from_vec(vec![scale * 1.2, scale * 0.9, scale * 1.1]).into_dyn(),
            ),
        ];

        let mut clipped_gradients = gradients.clone();
        let global_norm = clipper.clip_gradients(&mut clipped_gradients)?;

        let stats = clipper.get_statistics();
        println!(
            "  Epoch {}: norm={:.3}, adaptive_threshold={:.3}, clipped={}",
            epoch,
            global_norm,
            stats.adaptive_threshold,
            global_norm > stats.adaptive_threshold
        );
    }

    let final_stats = clipper.get_statistics();
    println!("\nFinal adaptive clipping statistics:");
    println!("  Total updates: {}", final_stats.total_updates);
    println!("  Times clipped: {}", final_stats.clip_count);
    println!(
        "  Clipping rate: {:.1}%",
        clipper.get_clipping_rate() * 100.0
    );
    println!(
        "  Final adaptive threshold: {:.3}",
        final_stats.adaptive_threshold
    );
    println!("  Average gradient norm: {:.3}", final_stats.avg_norm);

    Ok(())
}

/// Demonstrate parameter group clipping
fn parameter_group_clipping_demo() -> Result<()> {
    let mut clipper = GradientClipper::<f32>::new(GradientClippingConfig {
        max_norm: 1.0,
        per_parameter_clipping: true,
        ..Default::default()
    });

    // Configure different clipping thresholds for different parameter groups
    clipper.add_parameter_group("embeddings".to_string(), 0.5); // More aggressive
    clipper.add_parameter_group("transformer".to_string(), 1.0); // Standard
    clipper.add_parameter_group("output_head".to_string(), 2.0); // More lenient

    // Simulate gradients for different parameter groups
    let embedding_grads = vec![Tensor::from_array(
        Array1::from_vec(vec![1.2, 0.8, 1.0]).into_dyn(),
    )];
    let transformer_grads = vec![Tensor::from_array(
        Array1::from_vec(vec![1.5, 1.2, 0.9]).into_dyn(),
    )];
    let output_grads = vec![Tensor::from_array(
        Array1::from_vec(vec![2.5, 1.8, 2.2]).into_dyn(),
    )];

    println!("Parameter group clipping results:");

    let mut embedding_grads_mut = embedding_grads.clone();
    let embedding_norm = clipper.clip_parameter_group("embeddings", &mut embedding_grads_mut)?;
    println!(
        "  Embeddings (threshold=0.5): norm={:.3}, clipped={}",
        embedding_norm,
        embedding_norm > 0.5
    );

    let mut transformer_grads_mut = transformer_grads.clone();
    let transformer_norm =
        clipper.clip_parameter_group("transformer", &mut transformer_grads_mut)?;
    println!(
        "  Transformer (threshold=1.0): norm={:.3}, clipped={}",
        transformer_norm,
        transformer_norm > 1.0
    );

    let mut output_grads_mut = output_grads.clone();
    let output_norm = clipper.clip_parameter_group("output_head", &mut output_grads_mut)?;
    println!(
        "  Output head (threshold=2.0): norm={:.3}, clipped={}",
        output_norm,
        output_norm > 2.0
    );

    Ok(())
}

/// Demonstrate different norm types for gradient clipping
fn norm_types_comparison_demo() -> Result<()> {
    let test_gradients = vec![
        Tensor::from_array(Array1::from_vec(vec![1.0, 2.0, 3.0]).into_dyn()),
        Tensor::from_array(Array1::from_vec(vec![-1.5, 0.5, 2.5]).into_dyn()),
    ];

    // Test different norm types
    let norm_configs = vec![
        ("L1 Norm", NormType::L1, 5.0),
        ("L2 Norm", NormType::L2, 4.0),
        ("Infinity Norm", NormType::Infinity, 2.5),
    ];

    println!("Comparing different norm types:");

    for (name, norm_type, threshold) in norm_configs {
        let config = GradientClippingConfig {
            norm_type,
            max_norm: threshold,
            ..Default::default()
        };
        let mut clipper = GradientClipper::<f32>::new(config);

        let mut gradients_copy = test_gradients.clone();
        let computed_norm = clipper.clip_gradients(&mut gradients_copy)?;

        println!(
            "  {}: computed_norm={:.3}, threshold={:.1}, clipped={}",
            name,
            computed_norm,
            threshold,
            computed_norm > threshold
        );
    }

    Ok(())
}

/// Simulate a training loop with gradient clipping
fn training_simulation_demo() -> Result<()> {
    let mut clipper = GradientClipper::<f32>::new(GradientClippingConfig {
        max_norm: 1.0,
        adaptive_scaling: true,
        warmup_steps: 3,
        ..Default::default()
    });

    println!("Training simulation with warmup and adaptive scaling:");
    println!("Step | Grad Norm | Effective Threshold | Clipped | Clip Rate");
    println!("-----|-----------|-------------------|---------|----------");

    // Simulate unstable early training, then stabilization
    let gradient_patterns = vec![
        (3.0, "Early unstable"),
        (2.5, "Still unstable"),
        (2.0, "Stabilizing"),
        (1.5, "More stable"),
        (1.2, "Getting stable"),
        (0.8, "Stable"),
        (0.9, "Stable"),
        (1.1, "Occasional spike"),
        (0.7, "Stable"),
        (0.8, "Stable"),
    ];

    for (step, (base_norm, _phase)) in gradient_patterns.iter().enumerate() {
        // Create gradients with the target norm
        let target_elements = (base_norm / (2.0_f32).sqrt()) as f32; // For 2-element vector
        let gradients = vec![Tensor::from_array(
            Array1::from_vec(vec![target_elements, target_elements]).into_dyn(),
        )];

        let mut gradients_copy = gradients.clone();
        let actual_norm = clipper.clip_gradients(&mut gradients_copy)?;

        let stats = clipper.get_statistics();
        let config = clipper.get_config();
        let effective_threshold = if config.adaptive_scaling {
            stats.adaptive_threshold
        } else {
            config.max_norm
        };

        let was_clipped = actual_norm > effective_threshold;

        println!(
            "{:4} | {:9.3} | {:17.3} | {:7} | {:8.1}%",
            step + 1,
            actual_norm,
            effective_threshold,
            if was_clipped { "Yes" } else { "No" },
            clipper.get_clipping_rate() * 100.0
        );
    }

    println!("\nFinal training statistics:");
    let final_stats = clipper.get_statistics();
    println!("  Total gradient updates: {}", final_stats.total_updates);
    println!("  Times gradients were clipped: {}", final_stats.clip_count);
    println!(
        "  Overall clipping rate: {:.1}%",
        clipper.get_clipping_rate() * 100.0
    );
    println!(
        "  Final adaptive threshold: {:.3}",
        final_stats.adaptive_threshold
    );
    println!("  Average gradient norm: {:.3}", final_stats.avg_norm);
    println!("  Gradient norm std dev: {:.3}", final_stats.std_norm);

    Ok(())
}
