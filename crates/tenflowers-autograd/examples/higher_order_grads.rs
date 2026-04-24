//! Example: Higher-order gradients — third derivatives and nth-order derivatives.
//!
//! `tenflowers-autograd` supports arbitrary-order differentiation for scalar
//! functions via:
//!
//! * [`GradientTape::third_derivative`] — specialised 3rd-order path.
//! * [`GradientTape::nth_derivative`]   — general nth-order path (n ≥ 1).
//! * [`GradientTape::mixed_partial_derivative`] — mixed partials ∂^n f / ∂x_i ∂x_j …
//!
//! For second-order methods (Hessian, Jacobian, HVP) see
//! `second_order_derivatives_example.rs`.

use tenflowers_autograd::GradientTape;
use tenflowers_core::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Higher-order gradients example\n");

    // -----------------------------------------------------------------
    // 1. Third derivative of x³
    //    f(x) = x³   =>   f'''(x) = 6  (constant for all x)
    // -----------------------------------------------------------------
    println!("1. Third derivative of f(x) = x³ at x = 2.0");
    println!("   Expected f'''(2) = 6.0");

    let tape = GradientTape::new();
    let x_val = Tensor::<f32>::from_vec(vec![2.0_f32], &[1])?;
    let x = tape.watch(x_val);

    // Build x³ manually via the TrackedTensor API
    let x2 = x.mul(&x)?;
    let x3 = x2.mul(&x)?;

    let d3 = tape.third_derivative(&x3, &x)?;
    let val = d3.as_slice().map(|s| s[0]).unwrap_or(f32::NAN);
    println!("   Got: {val:.4} (note: numerical result may differ from analytical 6.0 due to FD approximation)\n");

    // -----------------------------------------------------------------
    // 2. 4th derivative of x⁴
    //    f(x) = x⁴   =>   f''''(x) = 24  (constant for all x)
    // -----------------------------------------------------------------
    println!("2. 4th derivative of f(x) = x⁴ at x = 1.0");
    println!("   Expected f''''(1) = 24.0");

    let tape = GradientTape::new();
    let x_val = Tensor::<f32>::from_vec(vec![1.0_f32], &[1])?;
    let x = tape.watch(x_val);

    let x2 = x.mul(&x)?;
    let x4 = x2.mul(&x2)?;

    let d4 = tape.nth_derivative(&x4, &x, 4)?;
    let val = d4.as_slice().map(|s| s[0]).unwrap_or(f32::NAN);
    println!("   Got: {val:.4}\n");

    // -----------------------------------------------------------------
    // 3. Mixed partial: ∂²f / ∂x∂y  for  f(x,y) = xy
    //    Expected: 1.0  (d/dx(d/dy xy) = d/dx(x) = 1)
    // -----------------------------------------------------------------
    println!("3. Mixed partial ∂²(xy)/∂x∂y at (x,y) = (1.0, 2.0)");
    println!("   Expected: 1.0");

    let tape = GradientTape::new();
    let x_val = Tensor::<f32>::from_vec(vec![1.0_f32], &[1])?;
    let y_val = Tensor::<f32>::from_vec(vec![2.0_f32], &[1])?;
    let x = tape.watch(x_val);
    let y = tape.watch(y_val);
    let xy = x.mul(&y)?;

    let mixed = tape.mixed_partial_derivative(&xy, &[&x, &y], &[1, 1])?;
    let val = mixed.as_slice().map(|s| s[0]).unwrap_or(f32::NAN);
    println!("   Got: {val:.4}\n");

    // -----------------------------------------------------------------
    // 4. nth_derivative for successive orders of f(x) = x⁵
    //    f'=5x⁴, f''=20x³, f'''=60x², f''''=120x, f'''''=120
    //    at x=1: [5, 20, 60, 120, 120]
    // -----------------------------------------------------------------
    println!("4. Successive derivatives of f(x) = x⁵ at x = 1.0");
    println!("   Analytical: [5, 20, 60, 120, 120]");
    print!("   Computed  : [");

    let tape = GradientTape::new();
    let x_val = Tensor::<f32>::from_vec(vec![1.0_f32], &[1])?;
    let x = tape.watch(x_val);
    let x2 = x.mul(&x)?;
    let x4 = x2.mul(&x2)?;
    let x5 = x4.mul(&x)?;

    for order in 1..=5_usize {
        let dn = tape.nth_derivative(&x5, &x, order)?;
        let val = dn.as_slice().map(|s| s[0]).unwrap_or(f32::NAN);
        if order < 5 {
            print!("{val:.0}, ");
        } else {
            println!("{val:.0}]");
        }
    }

    println!("\nHigher-order gradients example complete.");
    Ok(())
}
