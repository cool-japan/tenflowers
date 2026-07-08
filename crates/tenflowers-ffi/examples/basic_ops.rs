//! Demonstrates basic tensor creation and arithmetic using the TenfloweRS
//! FFI crate's internal Rust API.
//!
//! This example is intentionally self-contained: it uses only the types
//! re-exported by `tenflowers_ffi` without requiring a Python interpreter.
//!
//! Run with:
//!
//! ```text
//! cargo run --example basic_ops -p tenflowers-ffi
//! ```

use tenflowers_core::{ops, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== TenfloweRS: basic ops example ===\n");

    // ── Tensor creation ──────────────────────────────────────────────────────
    let zeros = Tensor::<f32>::zeros(&[2, 3]);
    let ones = Tensor::<f32>::ones(&[2, 3]);

    println!("zeros shape: {:?}", zeros.shape().dims());
    println!("ones  shape: {:?}", ones.shape().dims());

    // ── Arithmetic ───────────────────────────────────────────────────────────
    let sum = ops::add(&zeros, &ones)?;
    println!("\nzeros + ones =");
    print_tensor(&sum)?;

    let product = ops::mul(&ones, &ones)?;
    println!("\nones * ones =");
    print_tensor(&product)?;

    // ── From-vec construction ────────────────────────────────────────────────
    let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
    let b = Tensor::<f32>::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2])?;

    println!("\na =");
    print_tensor(&a)?;

    println!("\nb =");
    print_tensor(&b)?;

    let ab_sum = ops::add(&a, &b)?;
    println!("\na + b =");
    print_tensor(&ab_sum)?;

    // ── Matrix multiplication ────────────────────────────────────────────────
    let x = Tensor::<f32>::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2])?;
    let y = Tensor::<f32>::from_vec(vec![2.0, 3.0, 4.0, 5.0], &[2, 2])?;

    let matmul_result = ops::matmul(&x, &y)?;
    println!("\nidentity @ [[2,3],[4,5]] =");
    print_tensor(&matmul_result)?;

    println!("\nAll basic ops completed successfully.");
    Ok(())
}

/// Pretty-print a 2-D (or 1-D) tensor to stdout.
fn print_tensor(t: &Tensor<f32>) -> Result<(), Box<dyn std::error::Error>> {
    let data = t.to_vec()?;
    let dims = t.shape().dims().to_vec();

    match dims.len() {
        1 => {
            println!("  {:?}", data);
        }
        2 => {
            let cols = dims[1];
            for (i, chunk) in data.chunks(cols).enumerate() {
                let row: Vec<String> = chunk.iter().map(|v| format!("{:6.2}", v)).collect();
                println!("  row {}: [{}]", i, row.join(", "));
            }
        }
        _ => {
            println!("  <{}-D tensor, shape {:?}>", dims.len(), dims);
        }
    }

    Ok(())
}
