//! Demonstrates a manual SGD / Adam parameter-update loop using the tensor
//! operations from `tenflowers_core`.
//!
//! The optimizers in `tenflowers_neural` operate on `Model` objects; this
//! example instead implements the update rules manually using `ops`, which
//! is the approach best suited to standalone examples without a full model.
//!
//! Run with:
//!
//! ```text
//! cargo run --example optimizer_example -p tenflowers-ffi
//! ```

use tenflowers_core::{ops, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== TenfloweRS: optimizer example ===\n");

    // ── SGD: θ ← θ - lr · ∇θ ────────────────────────────────────────────────
    println!("--- Manual SGD (lr=0.1) ---");
    let lr = 0.1f32;

    let theta = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], &[3])?;
    let grad = Tensor::<f32>::from_vec(vec![0.5, -0.5, 1.0], &[3])?;

    println!("theta (initial): {:?}", theta.to_vec()?);
    println!("gradient:        {:?}", grad.to_vec()?);

    // Δθ = lr · grad
    let lr_tensor = Tensor::<f32>::from_scalar(lr);
    let delta = ops::mul(&lr_tensor, &grad)?;
    let theta_updated = ops::sub(&theta, &delta)?;

    println!("theta (after 1 SGD step): {:?}", theta_updated.to_vec()?);
    // Expected: [1.0 - 0.05, 2.0 + 0.05, 3.0 - 0.1] = [0.95, 2.05, 2.9]
    let updated_data = theta_updated.to_vec()?;
    assert!(
        (updated_data[0] - 0.95).abs() < 1e-5,
        "expected 0.95 got {}",
        updated_data[0]
    );
    assert!(
        (updated_data[1] - 2.05).abs() < 1e-5,
        "expected 2.05 got {}",
        updated_data[1]
    );
    assert!(
        (updated_data[2] - 2.90).abs() < 1e-5,
        "expected 2.90 got {}",
        updated_data[2]
    );

    // ── Multi-step SGD loop ───────────────────────────────────────────────────
    println!("\n--- Multi-step SGD (5 steps, constant gradient) ---");
    let mut p = Tensor::<f32>::from_vec(vec![1.0f32, 1.0], &[2])?;
    let g = Tensor::<f32>::from_vec(vec![0.5f32, -0.5], &[2])?;
    let lr_t = Tensor::<f32>::from_scalar(0.1f32);

    for step in 1..=5_usize {
        let update = ops::mul(&lr_t, &g)?;
        p = ops::sub(&p, &update)?;
        println!("  step {step}: {:?}", p.to_vec()?);
    }

    // After 5 steps: p[0] = 1.0 - 5*0.05 = 0.75; p[1] = 1.0 + 5*0.05 = 1.25
    let final_vals = p.to_vec()?;
    assert!(
        (final_vals[0] - 0.75).abs() < 1e-4,
        "expected p[0] ≈ 0.75, got {}",
        final_vals[0]
    );
    assert!(
        (final_vals[1] - 1.25).abs() < 1e-4,
        "expected p[1] ≈ 1.25, got {}",
        final_vals[1]
    );

    // ── Simplified Adam (one step): θ ← θ - lr·m̂/(√v̂+ε) ──────────────────
    println!("\n--- Simplified Adam (1 step) ---");
    let adam_lr = 0.001f32;
    let beta1 = 0.9f32;
    let beta2 = 0.999f32;
    let eps = 1e-8f32;

    let theta_adam = Tensor::<f32>::from_vec(vec![0.5f32, -0.5, 0.3], &[3])?;
    let grad_adam = Tensor::<f32>::from_vec(vec![0.1f32, 0.2, -0.1], &[3])?;

    // t=1, bias-corrected moments
    let m = Tensor::<f32>::zeros(&[3]); // first moment (initialised to 0)
    let v = Tensor::<f32>::zeros(&[3]); // second moment (initialised to 0)

    // m ← β1·m + (1-β1)·g
    let m_new = ops::add(
        &ops::mul(&Tensor::<f32>::from_scalar(beta1), &m)?,
        &ops::mul(&Tensor::<f32>::from_scalar(1.0 - beta1), &grad_adam)?,
    )?;

    // v ← β2·v + (1-β2)·g²
    let g_sq = ops::mul(&grad_adam, &grad_adam)?;
    let v_new = ops::add(
        &ops::mul(&Tensor::<f32>::from_scalar(beta2), &v)?,
        &ops::mul(&Tensor::<f32>::from_scalar(1.0 - beta2), &g_sq)?,
    )?;

    // Bias correction for t=1
    let m_hat_data: Vec<f32> = m_new.to_vec()?.iter().map(|&x| x / (1.0 - beta1)).collect();
    let v_hat_data: Vec<f32> = v_new.to_vec()?.iter().map(|&x| x / (1.0 - beta2)).collect();

    let m_hat = Tensor::<f32>::from_vec(m_hat_data, &[3])?;
    let v_hat = Tensor::<f32>::from_vec(v_hat_data, &[3])?;

    // denominator = sqrt(v_hat) + ε
    let sqrt_v_hat = ops::sqrt(&v_hat)?;
    let denom_data: Vec<f32> = sqrt_v_hat.to_vec()?.iter().map(|&x| x + eps).collect();
    let denom = Tensor::<f32>::from_vec(denom_data, &[3])?;

    // step = lr · m_hat / denom  (element-wise)
    let m_hat_data2 = m_hat.to_vec()?;
    let denom_data2 = denom.to_vec()?;
    let step_data: Vec<f32> = m_hat_data2
        .iter()
        .zip(denom_data2.iter())
        .map(|(&m, &d)| adam_lr * m / d)
        .collect();
    let step_tensor = Tensor::<f32>::from_vec(step_data, &[3])?;
    let theta_new_adam = ops::sub(&theta_adam, &step_tensor)?;

    println!("theta (before Adam step): {:?}", theta_adam.to_vec()?);
    println!("theta (after Adam step):  {:?}", theta_new_adam.to_vec()?);

    println!("\nAll optimizer examples completed successfully.");
    Ok(())
}
