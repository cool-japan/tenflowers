//! AdaBelief Optimizer Demonstration
//!
//! This example demonstrates how to use the new AdaBelief optimizer
//! for training neural networks with TenfloweRS.
//!
//! Usage:
//!   cargo run --example adabelief_demo

use std::time::Instant;
use tenflowers_core::{Device, Result, Tensor};
use tenflowers_neural::optimizers::{AdaBelief, AdaBeliefConfig};

fn main() -> Result<()> {
    println!("🚀 AdaBelief Optimizer Demonstration");
    println!("====================================\n");

    // Create AdaBelief optimizer with custom configuration
    println!("🔧 Creating AdaBelief optimizer with custom configuration...");
    let config = AdaBeliefConfig {
        lr: 0.01,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-16,
        weight_decay: 1e-4,
        amsgrad: true,
    };

    let mut optimizer = AdaBelief::<f32>::new(config);
    println!(
        "✅ Optimizer created with learning rate: {}",
        optimizer.get_lr()
    );

    // Simulate training a simple neural network layer
    println!("\n🧠 Simulating neural network parameter optimization...");

    // Create parameters for a simple 2x3 weight matrix
    let mut weights = Tensor::<f32>::from_vec(vec![0.1, -0.2, 0.3, 0.4, -0.1, 0.2], &[2, 3])?;

    // Create bias parameters
    let mut biases = Tensor::<f32>::from_vec(vec![0.0, 0.1, -0.05], &[3])?;

    // Register parameters with optimizer
    let weight_id = optimizer.register_param();
    let bias_id = optimizer.register_param();

    println!("📊 Initial parameters:");
    println!("   Weights: {:?}", weights.to_vec()?);
    println!("   Biases:  {:?}", biases.to_vec()?);

    // Simulate training for several steps
    println!("\n🏃 Running optimization steps...");
    let start_time = Instant::now();

    for step in 1..=20 {
        // Simulate gradients (in real training, these would come from backpropagation)
        let weight_gradients = Tensor::<f32>::from_vec(
            vec![
                0.02 * (step as f32).sin(),
                -0.01 * (step as f32).cos(),
                0.015,
                -0.01,
                0.005,
                -0.008,
            ],
            &[2, 3],
        )?;

        let bias_gradients = Tensor::<f32>::from_vec(
            vec![
                0.01 * (step as f32).sin(),
                -0.005,
                0.002 * (step as f32).cos(),
            ],
            &[3],
        )?;

        // Apply optimization steps
        optimizer.step(weight_id, &mut weights, &weight_gradients)?;
        optimizer.step(bias_id, &mut biases, &bias_gradients)?;

        // Print progress every 5 steps
        if step % 5 == 0 {
            let weight_norm = calculate_l2_norm(&weights)?;
            let bias_norm = calculate_l2_norm(&biases)?;
            println!(
                "   Step {}: Weight norm: {:.4}, Bias norm: {:.4}",
                step, weight_norm, bias_norm
            );
        }
    }

    let training_time = start_time.elapsed();
    println!("⏱️  Training completed in {:?}", training_time);

    // Show final parameters
    println!("\n📊 Final parameters after optimization:");
    println!("   Weights: {:?}", weights.to_vec()?);
    println!("   Biases:  {:?}", biases.to_vec()?);

    // Display optimizer state information
    println!("\n📈 Optimizer Statistics:");
    let state_info = optimizer.get_state_info();
    for (param_id, step_count) in state_info {
        let param_type = if param_id == weight_id {
            "Weights"
        } else {
            "Biases"
        };
        println!(
            "   {} (ID: {}): {} optimization steps",
            param_type, param_id, step_count
        );
    }

    // Demonstrate learning rate adjustment
    println!("\n🔧 Demonstrating learning rate adjustment...");
    let original_lr = optimizer.get_lr();
    optimizer.set_lr(original_lr * 0.5);
    println!(
        "   Original LR: {:.4} → New LR: {:.4}",
        original_lr,
        optimizer.get_lr()
    );

    // Compare with default AdaBelief
    println!("\n🆚 Comparing with default AdaBelief configuration:");
    let default_optimizer = AdaBelief::<f32>::default();
    println!(
        "   Custom config - LR: {:.4}, Beta1: {:.3}, Beta2: {:.3}, Eps: {:.0e}",
        optimizer.config().lr,
        optimizer.config().beta1,
        optimizer.config().beta2,
        optimizer.config().eps
    );
    println!(
        "   Default config - LR: {:.4}, Beta1: {:.3}, Beta2: {:.3}, Eps: {:.0e}",
        default_optimizer.config().lr,
        default_optimizer.config().beta1,
        default_optimizer.config().beta2,
        default_optimizer.config().eps
    );

    // Demonstrate convergence behavior
    println!("\n🎯 Demonstrating convergence on simple quadratic function...");
    demonstrate_quadratic_optimization()?;

    println!("\n✨ AdaBelief optimizer demonstration completed successfully!");
    println!("💡 Key advantages of AdaBelief:");
    println!("   • Adapts learning rate based on 'belief' in gradient predictions");
    println!("   • Better convergence properties than Adam in many scenarios");
    println!("   • Reduced overfitting compared to SGD with momentum");
    println!("   • Stable training for large language models");

    Ok(())
}

/// Calculate L2 norm of a f32 tensor
fn calculate_l2_norm(tensor: &Tensor<f32>) -> Result<f32> {
    let data = tensor.to_vec()?;
    let sum_of_squares: f32 = data.iter().map(|x| x * x).sum();
    Ok(sum_of_squares.sqrt())
}

/// Demonstrate AdaBelief optimization on a simple quadratic function
fn demonstrate_quadratic_optimization() -> Result<()> {
    println!("   Optimizing f(x) = (x - 2)^2, target: x = 2.0");

    let config = AdaBeliefConfig {
        lr: 0.1,
        ..AdaBeliefConfig::default()
    };
    let mut optimizer = AdaBelief::<f32>::new(config);
    let param_id = optimizer.register_param();

    // Start far from optimum
    let mut x = Tensor::<f32>::from_vec(vec![0.0], &[1])?;

    println!("   Initial x: {:.4}", x.to_vec()?[0]);

    for step in 1..=15 {
        // Gradient of f(x) = (x - 2)^2 is 2(x - 2)
        let x_val = x.to_vec()?[0];
        let grad_val = 2.0 * (x_val - 2.0);
        let gradient = Tensor::<f32>::from_vec(vec![grad_val], &[1])?;

        optimizer.step(param_id, &mut x, &gradient)?;

        let x_val = x.to_vec()?[0];
        let function_value = (x_val - 2.0) * (x_val - 2.0);

        if step % 3 == 0 {
            println!(
                "   Step {}: x = {:.4}, f(x) = {:.6}",
                step, x_val, function_value
            );
        }
    }

    let final_x = x.to_vec()?[0];
    let error = (final_x - 2.0).abs();
    println!("   Final result: x = {:.4}, error = {:.6}", final_x, error);

    if error < 0.1 {
        println!("   ✅ Successfully converged to the optimum!");
    } else {
        println!("   ⚠️  Convergence slower than expected, may need more iterations");
    }

    Ok(())
}
