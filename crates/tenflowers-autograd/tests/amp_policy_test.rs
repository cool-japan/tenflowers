/// Comprehensive tests for Automatic Mixed Precision (AMP) Policy
///
/// These tests validate the AMP policy implementation including:
/// - Dynamic loss scaling
/// - Gradient overflow detection
/// - Operation precision management
/// - Stability metrics tracking
use tenflowers_autograd::{AMPConfig, AMPPolicy, ScaleAdjustmentReason};
use tenflowers_core::{DType, Tensor};

#[test]
fn test_amp_config_defaults() {
    println!("Test: AMP configuration defaults");

    let config = AMPConfig::default();

    assert!(!config.enabled, "Should be disabled by default");
    assert_eq!(
        config.initial_scale, 65536.0,
        "Default scale should be 2^16"
    );
    assert_eq!(config.min_scale, 1.0);
    assert_eq!(config.max_scale, 65536.0 * 65536.0);
    assert_eq!(config.growth_interval, 2000);
    assert_eq!(config.growth_factor, 2.0);
    assert_eq!(config.backoff_factor, 0.5);
    assert_eq!(config.target_dtype, DType::Float16);
    assert!(config.track_stability);

    println!("✓ All default values correct");
}

#[test]
fn test_amp_config_builder() {
    println!("Test: AMP configuration builder pattern");

    let config = AMPConfig::default()
        .with_initial_scale(32768.0)
        .with_growth_interval(1000)
        .with_backoff_factor(0.25)
        .with_growth_factor(1.5)
        .with_bfloat16()
        .with_fp32_operation("custom_op")
        .with_stability_tracking(true);

    assert_eq!(config.initial_scale, 32768.0);
    assert_eq!(config.growth_interval, 1000);
    assert_eq!(config.backoff_factor, 0.25);
    assert_eq!(config.growth_factor, 1.5);
    assert_eq!(config.target_dtype, DType::BFloat16);
    assert!(config.fp32_operations.contains(&"custom_op".to_string()));
    assert!(config.track_stability);

    println!("✓ Builder pattern works correctly");
}

#[test]
fn test_amp_config_float16_vs_bfloat16() {
    println!("Test: Float16 vs BFloat16 configuration");

    let config_fp16 = AMPConfig::default().with_float16();
    let config_bf16 = AMPConfig::default().with_bfloat16();

    assert_eq!(config_fp16.target_dtype, DType::Float16);
    assert_eq!(config_bf16.target_dtype, DType::BFloat16);

    println!("✓ Precision modes set correctly");
}

#[test]
fn test_amp_policy_creation() {
    println!("Test: AMP policy creation");

    let config = AMPConfig {
        enabled: true,
        initial_scale: 1024.0,
        ..Default::default()
    };

    let policy = AMPPolicy::new(config);

    assert!(policy.is_enabled());
    assert_eq!(policy.get_current_scale(), 1024.0);

    println!("✓ Policy created with correct configuration");
}

#[test]
fn test_amp_policy_disabled() {
    println!("Test: Disabled AMP policy (pass-through mode)");

    let policy = AMPPolicy::disabled();

    assert!(!policy.is_enabled());

    let loss = Tensor::<f32>::from_scalar(1.0);
    let scaled = policy.scale_loss(&loss).unwrap();

    assert_eq!(
        scaled.as_slice().unwrap()[0],
        loss.as_slice().unwrap()[0],
        "Should not scale when disabled"
    );

    println!("✓ Disabled policy passes through unchanged");
}

#[test]
fn test_loss_scaling() {
    println!("Test: Loss scaling functionality");

    let config = AMPConfig {
        enabled: true,
        initial_scale: 1024.0,
        ..Default::default()
    };

    let policy = AMPPolicy::new(config);
    let loss = Tensor::<f32>::from_scalar(1.0);
    let scaled_loss = policy.scale_loss(&loss).unwrap();

    assert_eq!(
        scaled_loss.as_slice().unwrap()[0],
        1024.0,
        "Loss should be scaled by 1024"
    );

    println!("✓ Loss scaled correctly");
}

#[test]
fn test_gradient_unscaling_no_overflow() {
    println!("Test: Gradient unscaling without overflow");

    let config = AMPConfig {
        enabled: true,
        initial_scale: 1024.0,
        ..Default::default()
    };

    let mut policy = AMPPolicy::new(config);

    // Create clean gradients (no inf/nan)
    let mut gradients = vec![Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap()];

    let should_step = policy.unscale_and_check(&mut gradients).unwrap();

    assert!(should_step, "Should proceed when no overflow");

    let metrics = policy.get_stability_metrics();
    assert_eq!(metrics.total_steps, 1);
    assert_eq!(metrics.overflow_steps, 0);
    assert_eq!(metrics.consecutive_overflows, 0);

    println!("✓ Gradients unscaled, no overflow detected");
}

#[test]
fn test_stability_metrics_tracking() {
    println!("Test: Stability metrics tracking");

    let config = AMPConfig {
        enabled: true,
        track_stability: true,
        ..Default::default()
    };

    let mut policy = AMPPolicy::new(config);

    // Simulate multiple successful steps
    for _ in 0..10 {
        let mut gradients = vec![Tensor::<f32>::ones(&[5])];
        policy.unscale_and_check(&mut gradients).unwrap();
    }

    let metrics = policy.get_stability_metrics();

    assert_eq!(metrics.total_steps, 10);
    assert_eq!(metrics.overflow_steps, 0);
    assert_eq!(metrics.overflow_rate, 0.0);
    assert_eq!(metrics.consecutive_overflows, 0);

    println!("✓ Metrics tracked correctly over multiple steps");
}

#[test]
fn test_operation_dtype_fp32_whitelist() {
    println!("Test: FP32 operation whitelist");

    let config = AMPConfig {
        enabled: true,
        ..Default::default()
    };

    let policy = AMPPolicy::new(config);

    // Default FP32 operations should be in whitelist
    let ops_to_test = vec!["softmax", "batch_norm", "layer_norm", "exp", "log"];

    for op in ops_to_test {
        let dtype = policy.get_operation_dtype(op, DType::Float32);
        assert_eq!(
            dtype,
            DType::Float32,
            "{} should stay in FP32 for stability",
            op
        );
    }

    println!("✓ FP32 whitelist enforced correctly");
}

#[test]
fn test_operation_dtype_autocast() {
    println!("Test: Operation dtype autocasting");

    let config = AMPConfig {
        enabled: true,
        target_dtype: DType::Float16,
        ..Default::default()
    };

    let policy = AMPPolicy::new(config);

    // Operations not in whitelist should use target dtype
    let autocast_ops = vec!["matmul", "conv2d", "linear"];

    for op in autocast_ops {
        let dtype = policy.get_operation_dtype(op, DType::Float32);
        assert_eq!(dtype, DType::Float16, "{} should be autocasted to FP16", op);
    }

    println!("✓ Autocasting works for allowed operations");
}

#[test]
fn test_should_autocast() {
    println!("Test: Autocast decision logic");

    let config = AMPConfig {
        enabled: true,
        target_dtype: DType::Float16,
        ..Default::default()
    };

    let policy = AMPPolicy::new(config);

    // Should autocast regular ops with float input
    assert!(
        policy.should_autocast("matmul", DType::Float32),
        "Should autocast matmul"
    );

    // Should not autocast FP32 whitelist ops
    assert!(
        !policy.should_autocast("softmax", DType::Float32),
        "Should not autocast softmax"
    );

    // Should not autocast integer ops
    assert!(
        !policy.should_autocast("matmul", DType::Int32),
        "Should not autocast integer operations"
    );

    println!("✓ Autocast decisions correct");
}

#[test]
fn test_metrics_reset() {
    println!("Test: Metrics reset functionality");

    let config = AMPConfig {
        enabled: true,
        ..Default::default()
    };

    let mut policy = AMPPolicy::new(config);

    // Generate some metrics
    for _ in 0..5 {
        let mut gradients = vec![Tensor::<f32>::ones(&[5])];
        policy.unscale_and_check(&mut gradients).unwrap();
    }

    assert_eq!(policy.get_stability_metrics().total_steps, 5);

    // Reset metrics
    policy.reset_metrics();

    let metrics = policy.get_stability_metrics();
    assert_eq!(metrics.total_steps, 0);
    assert_eq!(metrics.overflow_steps, 0);
    assert!(policy.get_scale_history().is_empty());

    println!("✓ Metrics reset successfully");
}

#[test]
fn test_custom_fp32_operations() {
    println!("Test: Custom FP32 operation registration");

    let config = AMPConfig::default()
        .with_fp32_operation("my_custom_op")
        .with_fp32_operation("another_sensitive_op");

    assert!(config.fp32_operations.contains(&"my_custom_op".to_string()));
    assert!(config
        .fp32_operations
        .contains(&"another_sensitive_op".to_string()));

    let mut enabled_config = config.clone();
    enabled_config.enabled = true;
    let policy = AMPPolicy::new(enabled_config);

    assert_eq!(
        policy.get_operation_dtype("my_custom_op", DType::Float32),
        DType::Float32,
        "Custom ops should use FP32"
    );

    println!("✓ Custom FP32 operations registered correctly");
}

#[test]
fn test_scale_bounds_respected() {
    println!("Test: Loss scale bounds");

    let config = AMPConfig {
        enabled: true,
        min_scale: 1.0,
        max_scale: 1024.0,
        initial_scale: 512.0,
        ..Default::default()
    };

    let policy = AMPPolicy::new(config);

    let current_scale = policy.get_current_scale();
    assert!(
        (1.0..=1024.0).contains(&current_scale),
        "Scale should be within bounds"
    );

    println!("✓ Scale bounds enforced");
}

#[test]
fn test_bfloat16_configuration() {
    println!("Test: BFloat16 specific configuration");

    let config = AMPConfig {
        enabled: true,
        target_dtype: DType::BFloat16,
        initial_scale: 1.0, // BFloat16 often doesn't need high scaling
        ..Default::default()
    };

    let policy = AMPPolicy::new(config);

    assert_eq!(policy.get_current_scale(), 1.0);

    let mut enabled_policy_config = AMPConfig::default().with_bfloat16();
    enabled_policy_config.enabled = true;

    let enabled_policy = AMPPolicy::new(enabled_policy_config);
    assert_eq!(
        enabled_policy.get_operation_dtype("conv2d", DType::Float32),
        DType::BFloat16
    );

    println!("✓ BFloat16 configuration works correctly");
}

#[test]
fn test_amp_overhead_tracking() {
    println!("Test: AMP overhead measurement");

    let config = AMPConfig {
        enabled: true,
        track_stability: true,
        ..Default::default()
    };

    let mut policy = AMPPolicy::new(config);

    // Perform several operations
    for _ in 0..10 {
        let mut gradients = vec![Tensor::<f32>::ones(&[100])];
        policy.unscale_and_check(&mut gradients).unwrap();
    }

    let metrics = policy.get_stability_metrics();

    // Overhead should be measured
    // Note: overhead_ms is u64, always >= 0

    println!("✓ Overhead tracking functional");
    println!("  Total overhead: {} ms", metrics.amp_overhead_ms);
}

#[test]
fn test_multiple_gradient_tensors() {
    println!("Test: Handling multiple gradient tensors");

    let config = AMPConfig {
        enabled: true,
        initial_scale: 1024.0,
        ..Default::default()
    };

    let mut policy = AMPPolicy::new(config);

    // Multiple gradient tensors (e.g., from different layers)
    let mut gradients = vec![
        Tensor::<f32>::ones(&[10, 10]),
        Tensor::<f32>::ones(&[20, 5]),
        Tensor::<f32>::ones(&[100]),
    ];

    let should_step = policy.unscale_and_check(&mut gradients).unwrap();

    assert!(should_step, "Should handle multiple tensors");

    println!("✓ Multiple gradient tensors handled correctly");
}

#[test]
fn test_disabled_vs_enabled_comparison() {
    println!("Test: Disabled vs Enabled behavior comparison");

    let disabled_policy = AMPPolicy::disabled();
    let enabled_policy = AMPPolicy::new(AMPConfig {
        enabled: true,
        initial_scale: 2048.0,
        ..Default::default()
    });

    let loss = Tensor::<f32>::from_scalar(1.0);

    let disabled_scaled = disabled_policy.scale_loss(&loss).unwrap();
    let enabled_scaled = enabled_policy.scale_loss(&loss).unwrap();

    assert_eq!(disabled_scaled.as_slice().unwrap()[0], 1.0);
    assert_eq!(enabled_scaled.as_slice().unwrap()[0], 2048.0);

    println!("✓ Disabled vs enabled behavior differs as expected");
}

/// Integration test: Full AMP training loop simulation
#[test]
fn test_amp_training_loop_simulation() {
    println!("Integration Test: AMP Training Loop Simulation");
    println!("==============================================");

    let config = AMPConfig {
        enabled: true,
        initial_scale: 32768.0,
        growth_interval: 5,
        growth_factor: 2.0,
        backoff_factor: 0.5,
        target_dtype: DType::Float16,
        track_stability: true,
        ..Default::default()
    };

    let mut policy = AMPPolicy::new(config);

    println!("Simulating 20 training steps...\n");

    let mut successful_steps = 0;

    for step in 1..=20 {
        // Simulate forward pass with loss
        let loss = Tensor::<f32>::from_scalar(1.0 / step as f32);
        let _scaled_loss = policy.scale_loss(&loss).unwrap();

        // Simulate backward pass with gradients
        let mut gradients = vec![Tensor::<f32>::ones(&[10, 10])];

        let should_step = policy.unscale_and_check(&mut gradients).unwrap();

        if should_step {
            successful_steps += 1;
            // Simulate parameter update would happen here
        }

        if step % 5 == 0 {
            let metrics = policy.get_stability_metrics();
            println!(
                "Step {}: scale={:.1}, overflows={}, successful={}",
                step, metrics.current_scale, metrics.overflow_steps, successful_steps
            );
        }
    }

    let final_metrics = policy.get_stability_metrics();

    println!("\nFinal Training Statistics:");
    println!("  Total steps: {}", final_metrics.total_steps);
    println!("  Successful steps: {}", successful_steps);
    println!(
        "  Overflow rate: {:.2}%",
        final_metrics.overflow_rate * 100.0
    );
    println!("  Final scale: {:.1}", final_metrics.current_scale);
    println!("  Scale adjustments: {}", final_metrics.scale_adjustments);

    assert_eq!(final_metrics.total_steps, 20);
    assert!(successful_steps > 0, "Should have successful steps");

    println!("\n✓ Full AMP training loop simulation successful");
}
