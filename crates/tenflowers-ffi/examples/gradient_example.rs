//! Demonstrates forward / backward gradient checking using the
//! [`GradientParityChecker`] from `tenflowers_ffi::gradient_parity`.
//!
//! The checker compares analytically-provided gradients against central
//! finite-difference approximations and reports whether they match within
//! the configured tolerances.
//!
//! Run with:
//!
//! ```text
//! cargo run --example gradient_example -p tenflowers-ffi
//! ```

use tenflowers::{GradientParityChecker, GradientParityResult};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== TenfloweRS: gradient parity example ===\n");

    // ── Example 1: f(x) = sum(x²), ∇f = 2x ─────────────────────────────────
    println!("--- Example 1: f(x) = sum(x²) ---");
    let checker = GradientParityChecker::new(1e-5, 1e-4);
    let input = [1.0f32, 2.0f32, 3.0f32];
    let analytical_grad = [2.0f32, 4.0f32, 6.0f32]; // 2·x

    let result = checker.check_scalar_function(
        |x| x.iter().map(|&v| v * v).sum::<f32>(),
        &input,
        &analytical_grad,
    )?;

    print_result("sum-of-squares", &result);

    // ── Example 2: f(x) = sum(x³), ∇f = 3x² ────────────────────────────────
    println!("\n--- Example 2: f(x) = sum(x³) ---");
    let input3 = [1.0f32, 2.0f32];
    let analytical_grad3 = [3.0f32, 12.0f32]; // 3·x²

    let result3 = checker.check_scalar_function(
        |x| x.iter().map(|&v| v * v * v).sum::<f32>(),
        &input3,
        &analytical_grad3,
    )?;

    print_result("sum-of-cubes", &result3);

    // ── Example 3: f(x) = dot(a, x), ∇f = a  ───────────────────────────────
    println!("\n--- Example 3: f(x) = dot(a, x), a = [1,2,3] ---");
    let a = [1.0f32, 2.0f32, 3.0f32];
    let input_dot = [0.5f32, 1.0f32, 1.5f32];
    let analytical_grad_dot = a; // gradient of dot product w.r.t. x is a

    let result_dot = checker.check_scalar_function(
        |x| {
            x.iter()
                .zip(a.iter())
                .map(|(&xi, &ai)| xi * ai)
                .sum::<f32>()
        },
        &input_dot,
        &analytical_grad_dot,
    )?;

    print_result("dot-product", &result_dot);

    // ── Summary ──────────────────────────────────────────────────────────────
    println!("\n=== Summary ===");
    let all_pass = result.is_passing() && result3.is_passing() && result_dot.is_passing();
    if all_pass {
        println!("All gradient checks passed.");
    } else {
        eprintln!("One or more gradient checks failed — see individual reports above.");
        std::process::exit(1);
    }

    Ok(())
}

fn print_result(label: &str, result: &GradientParityResult) {
    let status = if result.is_passing() { "PASS" } else { "FAIL" };
    println!(
        "[{}] {} | max_abs={:.2e} max_rel={:.2e} rms={:.2e}",
        status, label, result.max_abs_error, result.max_rel_error, result.rms_error,
    );
    if !result.is_passing() {
        println!("  Full report:\n{}", result.format_report());
    }
}
