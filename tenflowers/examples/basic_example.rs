//! Basic tensor creation and arithmetic using the `tenflowers` meta-crate prelude.
//!
//! This example demonstrates the most common entry points available after a
//! single `use tenflowers::prelude::*;` import.
//!
//! Run with:
//!
//! ```text
//! cargo run --example basic_example -p tenflowers
//! ```

use tenflowers::prelude::*;

fn main() -> Result<()> {
    println!("=== TenfloweRS: basic example ===\n");
    println!("framework version: {}", tenflowers::VERSION);

    // ── Tensor creation ──────────────────────────────────────────────────────
    let zeros = Tensor::<f32>::zeros(&[3, 4]);
    let ones = Tensor::<f32>::ones(&[3, 4]);

    println!("zeros shape: {:?}", zeros.shape().dims());
    println!("ones  shape: {:?}", ones.shape().dims());

    // ── Arithmetic operations ────────────────────────────────────────────────
    let sum = ops::add(&zeros, &ones)?;
    println!("\nzeros + ones element count: {}", sum.size());

    let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
    let b = Tensor::<f32>::from_vec(vec![10.0, 20.0, 30.0, 40.0], &[2, 2])?;
    let ab = ops::add(&a, &b)?;

    println!("\na + b = {:?}", ab.to_vec()?);

    // ── Matrix multiplication ────────────────────────────────────────────────
    // identity × b = b
    let eye = Tensor::<f32>::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2])?;
    let product = ops::matmul(&eye, &b)?;
    println!("\nidentity @ b = {:?}", product.to_vec()?);

    // ── Type alias usage ─────────────────────────────────────────────────────
    let _vec: Vector = Tensor::<f32>::zeros(&[8]);
    let _mat: Matrix = Tensor::<f32>::zeros(&[4, 4]);

    // ── Reduction ────────────────────────────────────────────────────────────
    let mean_val = ops::mean(&ones, None, false)?;
    println!("\nmean(ones) = {:?}", mean_val.to_vec()?);

    println!("\nBasic example completed successfully.");
    Ok(())
}
