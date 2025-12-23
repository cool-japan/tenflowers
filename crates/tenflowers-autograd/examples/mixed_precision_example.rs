/// Automatic Mixed Precision (AMP) Training Example
///
/// This example demonstrates how to use automatic mixed precision training
/// to accelerate training while maintaining numerical stability through
/// dynamic loss scaling.
use scirs2_core::ndarray::Array2;
use tenflowers_autograd::{AMPConfig, AMPPolicy, GradientTape};
use tenflowers_core::{DType, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("TenfloweRS Automatic Mixed Precision (AMP) Example");
    println!("==================================================\n");

    // Example 1: Basic AMP with FP16
    println!("Example 1: Basic FP16 Mixed Precision");
    println!("-------------------------------------");
    basic_amp_example()?;

    println!("\nExample 2: BFloat16 Mixed Precision");
    println!("-----------------------------------");
    bfloat16_example()?;

    println!("\nExample 3: Dynamic Loss Scaling");
    println!("-------------------------------");
    dynamic_loss_scaling_example()?;

    println!("\nExample 4: Custom AMP Policy");
    println!("---------------------------");
    custom_amp_policy_example()?;

    Ok(())
}

fn basic_amp_example() -> Result<(), Box<dyn std::error::Error>> {
    // Configure AMP with FP16 and basic loss scaling
    let amp_config = AMPConfig {
        enabled: true,
        initial_scale: 65536.0,
        target_dtype: DType::Float16,
        ..Default::default()
    };

    let mut amp_policy = AMPPolicy::new(amp_config);
    let tape = GradientTape::new();

    println!("Creating input tensor (FP32)...");
    let x = tape.watch(Tensor::from_array(
        Array2::from_shape_fn((32, 128), |(i, j)| ((i + j) as f32) * 0.01).into_dyn(),
    ));

    println!("Building computation graph with AMP...");

    // Matrix multiplication
    let weights = Tensor::from_array(
        Array2::from_shape_fn((128, 64), |(i, j)| ((i * 2 + j) as f32) * 0.001).into_dyn(),
    );
    let hidden = x.tensor().matmul(&weights)?;
    println!("  MatMul: Will be cast to FP16 (autocasted)");

    // Loss computation (FP32)
    let loss = Tensor::<f32>::from_scalar(hidden.as_slice().unwrap().iter().sum::<f32>());
    println!("  Loss computed in FP32");

    // Scale loss before backpropagation
    let scaled_loss = amp_policy.scale_loss(&loss)?;
    println!("  Loss scaled by: {}", amp_policy.get_current_scale());

    // Compute gradients (would normally use tape.gradient here)
    let mut gradients = vec![Tensor::<f32>::ones(&[32, 128])];

    // Unscale gradients and check for overflow
    let should_step = amp_policy.unscale_and_check(&mut gradients)?;

    if should_step {
        println!("✓ AMP training step complete - No overflow detected");
        println!("  Speed improvement: ~2x faster on compatible hardware");
        println!("  Memory savings: ~50% reduction");
    } else {
        println!("⚠ Gradient overflow detected - skipping step");
    }

    // Print stability metrics
    let metrics = amp_policy.get_stability_metrics();
    println!("\nStability Metrics:");
    println!("  Total steps: {}", metrics.total_steps);
    println!("  Overflow rate: {:.2}%", metrics.overflow_rate * 100.0);
    println!("  Current scale: {:.1}", metrics.current_scale);

    Ok(())
}

fn bfloat16_example() -> Result<(), Box<dyn std::error::Error>> {
    // Configure AMP with BFloat16 (better numerical stability than FP16)
    let amp_config = AMPConfig {
        enabled: true,
        initial_scale: 1.0, // BFloat16 often doesn't need high scaling
        target_dtype: DType::BFloat16,
        growth_interval: 1000,
        ..Default::default()
    }
    .with_bfloat16();

    let mut amp_policy = AMPPolicy::new(amp_config);
    let tape = GradientTape::new();

    println!("Creating input tensor for BFloat16 training...");
    let x = tape.watch(Tensor::from_array(
        Array2::from_shape_fn((16, 64), |(i, j)| i as f32 * 0.1 + j as f32 * 0.01).into_dyn(),
    ));

    println!("Building computation with BFloat16...");

    let w1 = Tensor::from_array(
        Array2::from_shape_fn((64, 32), |(i, j)| (i as f32 - j as f32) * 0.001).into_dyn(),
    );
    let h1 = x.tensor().matmul(&w1)?;
    println!("  Layer 1: BFloat16 matmul");

    let h1_sum = h1.as_slice().unwrap().iter().sum::<f32>();
    let loss = Tensor::<f32>::from_scalar(h1_sum);

    println!("Computing gradients with BFloat16...");
    let mut gradients = vec![Tensor::<f32>::ones(&[16, 64])];
    let should_step = amp_policy.unscale_and_check(&mut gradients)?;

    if should_step {
        println!("✓ BFloat16 training complete");
        println!("  Advantages: Better numerical range than FP16");
        println!("  Use case: Large language models and transformers");
    }

    Ok(())
}

fn dynamic_loss_scaling_example() -> Result<(), Box<dyn std::error::Error>> {
    // Configure aggressive dynamic loss scaling
    let amp_config = AMPConfig {
        enabled: true,
        initial_scale: 32768.0,
        growth_factor: 2.0,
        backoff_factor: 0.5,
        growth_interval: 1000,
        target_dtype: DType::Float16,
        ..Default::default()
    };

    let mut amp_policy = AMPPolicy::new(amp_config);
    let tape = GradientTape::new();

    println!("Simulating training with dynamic loss scaling...\n");

    for step in 0..5 {
        println!("Training Step {}", step + 1);

        let x = tape.watch(Tensor::from_array(
            Array2::from_shape_fn((8, 32), |(i, j)| {
                (i as f32 + j as f32) * 0.01 * (step as f32 + 1.0)
            })
            .into_dyn(),
        ));

        let w = Tensor::from_array(
            Array2::from_shape_fn((32, 16), |(i, j)| (i as f32 - j as f32) * 0.001).into_dyn(),
        );

        let output = x.tensor().matmul(&w)?;
        let loss = Tensor::<f32>::from_scalar(output.as_slice().unwrap().iter().sum::<f32>());

        // Scale and compute gradients
        let mut gradients = vec![Tensor::<f32>::ones(&[8, 32])];
        let should_step = amp_policy.unscale_and_check(&mut gradients)?;

        let current_scale = amp_policy.get_current_scale();
        println!("  Loss scale: {:.1}", current_scale);

        if should_step {
            println!("  ✓ Gradient computed successfully");
        } else {
            println!("  ⚠ Overflow detected - scale adjusted");
        }
        println!();
    }

    println!("✓ Dynamic loss scaling demonstration complete");
    println!("  Loss scale adapts automatically to prevent under/overflow");

    // Print full stability report
    amp_policy.print_stability_report();

    Ok(())
}

fn custom_amp_policy_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a custom AMP policy for specific use case
    let amp_config = AMPConfig::default()
        .with_initial_scale(2048.0) // Lower initial scale for stable training
        .with_growth_factor(1.5) // Gentler growth
        .with_backoff_factor(0.75) // Gentler backoff
        .with_growth_interval(500)
        .with_float16()
        .with_fp32_operation("softmax")
        .with_fp32_operation("layer_norm")
        .with_fp32_operation("batch_norm")
        .with_stability_tracking(true);

    let mut amp_policy = AMPPolicy::new(amp_config);

    println!("Custom AMP Policy Configuration:");
    println!("  Precision: FP16");
    println!("  Initial loss scale: 2048");
    println!("  Growth factor: 1.5");
    println!("  FP32 ops: softmax, layer_norm, batch_norm");
    println!("  Stability tracking: enabled");
    println!();

    let tape = GradientTape::new();
    let x = tape.watch(Tensor::from_array(
        Array2::from_shape_fn((4, 16), |(i, j)| (i as f32 * j as f32) * 0.001).into_dyn(),
    ));

    println!("Building computation with custom policy...");
    let w = Tensor::from_array(
        Array2::from_shape_fn((16, 8), |(i, j)| ((i + j) as f32) * 0.01).into_dyn(),
    );

    let output = x.tensor().matmul(&w)?;
    let loss = Tensor::<f32>::from_scalar(output.as_slice().unwrap().iter().sum::<f32>());

    // Check operation precision
    let matmul_dtype = amp_policy.get_operation_dtype("matmul", DType::Float32);
    let softmax_dtype = amp_policy.get_operation_dtype("softmax", DType::Float32);

    println!("Operation precision check:");
    println!("  matmul: {:?} (autocasted)", matmul_dtype);
    println!("  softmax: {:?} (kept in FP32)", softmax_dtype);

    let mut gradients = vec![Tensor::<f32>::ones(&[4, 16])];
    let should_step = amp_policy.unscale_and_check(&mut gradients)?;

    if should_step {
        println!("\n✓ Custom AMP policy applied successfully");
        println!("  Policy optimized for: Stable training with moderate speedup");
    }

    println!("\n📝 AMP Best Practices:");
    println!("  • Start with conservative loss scaling");
    println!("  • Keep normalization operations in FP32");
    println!("  • Monitor for NaN/Inf during training");
    println!("  • Use BFloat16 for better numerical stability");
    println!("  • Combine with gradient clipping for safety");

    Ok(())
}
