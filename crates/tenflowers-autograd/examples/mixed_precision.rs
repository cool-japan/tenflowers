//! Example: Mixed-precision gradient computation using AMPPolicy.
//!
//! This example shows how to embed `AMPPolicy` into a manual training loop
//! that tracks loss-scaling statistics across multiple steps. The operation-
//! level precision policy selects BF16 for compute-heavy ops while keeping
//! numerically sensitive ops (softmax, loss) in FP32.
//!
//! For a more comprehensive AMP walkthrough, see `mixed_precision_example.rs`.

use tenflowers_autograd::{AMPConfig, AMPPolicy, GradientTape};
use tenflowers_core::{DType, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Mixed-precision gradient computation — operation-level policy demo\n");

    // --- 1. Configure the AMP policy -----------------------------------------
    //
    // BFloat16 preserves the FP32 dynamic range while halving storage; it is
    // preferred over FP16 for LLM/transformer fine-tuning. Numerically sensitive
    // ops (softmax, cross-entropy) stay in FP32 automatically.
    let amp_config = AMPConfig::default()
        .with_initial_scale(65_536.0)
        .with_growth_interval(2_000)
        .with_backoff_factor(0.5)
        .with_growth_factor(2.0)
        .with_bfloat16()
        .with_stability_tracking(true)
        // Always keep these ops in FP32 (softmax, batch-norm stats, etc.)
        .with_fp32_operation("softmax".to_string())
        .with_fp32_operation("layer_norm".to_string())
        .with_fp32_operation("cross_entropy".to_string());

    let mut amp_policy = AMPPolicy::new(amp_config);

    // --- 2. Simulated training loop -------------------------------------------
    //
    // A real loop would use a DataLoader and a proper model; here we use
    // synthetic tensors to demonstrate the loss-scaling bookkeeping.
    let num_steps = 6;

    for step in 0..num_steps {
        let tape = GradientTape::new();

        // Synthetic input: batch_size=8, features=32
        let _x = tape.watch(Tensor::<f32>::ones(&[8, 32]));

        // Simulate a scalar loss (in practice: call model.forward + loss fn)
        let loss = Tensor::<f32>::from_vec(vec![0.5_f32 * (1.0 + step as f32 * 0.1)], &[1])?;

        // Check whether AMP should autocast this operation to BF16
        let _cast_matmul = amp_policy.should_autocast("matmul", DType::Float32);
        let _cast_softmax = amp_policy.should_autocast("softmax", DType::Float32);

        // Scale loss before backward
        let _scaled = amp_policy.scale_loss(&loss)?;

        // Unscale gradients (simulate backward pass result)
        let mut grads = vec![Tensor::<f32>::ones(&[8, 32])];
        let step_ok = amp_policy.unscale_and_check(&mut grads)?;

        let scale = amp_policy.get_current_scale();
        let loss_val = loss.as_slice().map(|s| s[0]).unwrap_or(0.0);
        if step_ok {
            println!("step {step}: loss={loss_val:.3}  scale={scale:.0}  status=ok");
        } else {
            println!("step {step}: loss={loss_val:.3}  scale={scale:.0}  status=skip(overflow)");
        }
    }

    // --- 3. Stability report --------------------------------------------------
    let metrics = amp_policy.get_stability_metrics();
    println!("\nAMP summary after {num_steps} steps:");
    println!("  overflow rate : {:.1}%", metrics.overflow_rate * 100.0);
    println!("  current scale : {:.0}", metrics.current_scale);
    println!("  total steps   : {}", metrics.total_steps);

    Ok(())
}
