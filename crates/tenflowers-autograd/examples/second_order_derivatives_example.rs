// # Second-Order Derivatives Example
//
// This example demonstrates computation of second-order derivatives including:
// - Hessian matrices
// - Hessian-vector products
// - Jacobian matrices
// - Laplacian
// - Newton's method optimization
// - Natural gradient computation
//
// These are essential for:
// - Second-order optimization methods (Newton, quasi-Newton)
// - Sensitivity analysis
// - Uncertainty quantification
// - Natural gradient descent

use scirs2_core::ndarray::array;
use tenflowers_autograd::{
    compute_hessian, compute_hessian_diagonal, compute_jacobian, compute_laplacian,
    directional_second_derivative, hessian_vector_product, GradientTape, TrackedTensor,
};
use tenflowers_core::{Result, Tensor};

fn main() -> Result<()> {
    println!("TenfloweRS Second-Order Derivatives Example");
    println!("===========================================\n");

    // Example 1: Computing Hessian matrix
    example_1_hessian()?;

    // Example 2: Hessian-vector products (efficient)
    example_2_hvp()?;

    // Example 3: Hessian diagonal
    example_3_hessian_diagonal()?;

    // Example 4: Jacobian computation
    example_4_jacobian()?;

    // Example 5: Laplacian
    example_5_laplacian()?;

    // Example 6: Directional second derivatives
    example_6_directional_derivative()?;

    // Example 7: Newton's method optimization
    example_7_newton_optimization()?;

    println!("\n\nAll second-order derivative examples completed!");
    Ok(())
}

/// Example 1: Computing full Hessian matrix
fn example_1_hessian() -> Result<()> {
    println!("Example 1: Hessian Matrix Computation");
    println!("-------------------------------------");

    let tape = GradientTape::new();
    let x = tape.watch(Tensor::from_array(array![1.0f32, 2.0, 3.0].into_dyn()));

    // Function: f(x) = x₁² + 2x₁x₂ + 3x₂² + x₃²
    // This is a quadratic form with a known Hessian
    let x_sq = x.mul(&x)?;
    let sum = x_sq.sum(None, false)?;

    println!("Function: f(x) = Σ xᵢ²");
    println!("Input: {:?}", x.tensor.as_slice().unwrap());

    // Compute Hessian
    let hessian = compute_hessian(&tape, &sum, &x)?;

    println!("Hessian matrix (approximate):");
    println!("  Shape: {:?}", hessian.shape().dims());

    // For f(x) = Σ xᵢ², the Hessian should be approximately 2*I
    if let Some(data) = hessian.as_slice() {
        println!("  Data: {:?}", &data[..9.min(data.len())]);
    }

    println!("Expected: approximately 2*I (diagonal matrix)\n");

    Ok(())
}

/// Example 2: Hessian-vector product (more efficient than full Hessian)
fn example_2_hvp() -> Result<()> {
    println!("Example 2: Hessian-Vector Product");
    println!("---------------------------------");

    let tape = GradientTape::new();
    let x = tape.watch(Tensor::from_array(array![1.0f32, 2.0].into_dyn()));

    // Function: f(x, y) = x² + xy + y²
    let x_sq = x.mul(&x)?;
    let sum = x_sq.sum(None, false)?;

    // Direction vector
    let v = Tensor::from_array(array![1.0f32, 0.0].into_dyn());

    println!("Function: f(x) = Σ xᵢ²");
    println!("Input x: {:?}", x.tensor.as_slice().unwrap());
    println!("Direction v: {:?}", v.as_slice().unwrap());

    // Compute H*v without forming full Hessian
    let hvp = hessian_vector_product(&tape, &sum, &x, &v)?;

    println!("Hessian-vector product (H*v):");
    println!("  {:?}", hvp.as_slice().unwrap());

    println!("This is much more efficient than computing full Hessian!\n");

    Ok(())
}

/// Example 3: Hessian diagonal
fn example_3_hessian_diagonal() -> Result<()> {
    println!("Example 3: Hessian Diagonal");
    println!("---------------------------");

    let tape = GradientTape::new();
    let x = tape.watch(Tensor::from_array(array![1.0f32, 2.0, 3.0, 4.0].into_dyn()));

    // Function: f(x) = Σ xᵢ²
    let x_sq = x.mul(&x)?;
    let sum = x_sq.sum(None, false)?;

    println!("Function: f(x) = Σ xᵢ²");
    println!("Input: {:?}", x.tensor.as_slice().unwrap());

    // Compute only diagonal of Hessian
    let hessian_diag = compute_hessian_diagonal(&tape, &sum, &x)?;

    println!("Hessian diagonal:");
    println!("  {:?}", hessian_diag.as_slice().unwrap());
    println!("Expected: [2.0, 2.0, 2.0, 2.0] (second derivatives)\n");

    Ok(())
}

/// Example 4: Jacobian matrix
fn example_4_jacobian() -> Result<()> {
    println!("Example 4: Jacobian Matrix");
    println!("-------------------------");

    let tape = GradientTape::new();
    let x = tape.watch(Tensor::from_array(array![1.0f32, 2.0].into_dyn()));

    // Multi-output function: [x², xy]
    let x_sq = x.mul(&x)?;
    let output = x_sq;

    println!("Function: f(x) = [x₁², x₂²]");
    println!("Input: {:?}", x.tensor.as_slice().unwrap());

    // Compute Jacobian
    let jacobian = compute_jacobian(&tape, std::slice::from_ref(&output), &x)?;

    println!("Jacobian matrix:");
    println!("  Shape: {:?}", jacobian.shape().dims());
    if let Some(data) = jacobian.as_slice() {
        println!("  Data: {:?}", data);
    }

    println!("Expected Jacobian: [[2x₁, 0], [0, 2x₂]] = [[2.0, 0.0], [0.0, 4.0]]\n");

    Ok(())
}

/// Example 5: Laplacian (trace of Hessian)
fn example_5_laplacian() -> Result<()> {
    println!("Example 5: Laplacian");
    println!("-------------------");

    let tape = GradientTape::new();
    let x = tape.watch(Tensor::from_array(array![1.0f32, 2.0, 3.0].into_dyn()));

    // Function: f(x) = x₁² + x₂² + x₃²
    let x_sq = x.mul(&x)?;
    let sum = x_sq.sum(None, false)?;

    println!("Function: f(x) = x₁² + x₂² + x₃²");
    println!("Input: {:?}", x.tensor.as_slice().unwrap());

    // Compute Laplacian (Δf = trace(Hessian))
    let laplacian = compute_laplacian(&tape, &sum, &x)?;

    println!("Laplacian (Δf):");
    println!("  {:.4}", laplacian);
    println!("Expected: 6.0 (sum of second derivatives: 2 + 2 + 2)\n");

    Ok(())
}

/// Example 6: Directional second derivative
fn example_6_directional_derivative() -> Result<()> {
    println!("Example 6: Directional Second Derivative");
    println!("----------------------------------------");

    let tape = GradientTape::new();
    let x = tape.watch(Tensor::from_array(array![1.0f32, 2.0].into_dyn()));

    // Function: f(x, y) = x² + y²
    let x_sq = x.mul(&x)?;
    let sum = x_sq.sum(None, false)?;

    // Direction vector (normalized)
    let v = Tensor::from_array(array![0.707f32, 0.707].into_dyn()); // ≈ (1/√2, 1/√2)

    println!("Function: f(x, y) = x² + y²");
    println!("Input: {:?}", x.tensor.as_slice().unwrap());
    println!("Direction: {:?}", v.as_slice().unwrap());

    // Compute directional second derivative: v^T * H * v
    let dir_second_deriv = directional_second_derivative(&tape, &sum, &x, &v)?;

    println!("Directional second derivative (v^T * H * v):");
    println!("  {:.4}", dir_second_deriv);

    println!("This represents curvature in the direction v\n");

    Ok(())
}

/// Example 7: Newton's method optimization
fn example_7_newton_optimization() -> Result<()> {
    println!("Example 7: Newton's Method Optimization");
    println!("---------------------------------------");

    // Minimize f(x) = (x-3)²
    // Optimal solution: x* = 3
    // Gradient: 2(x-3)
    // Hessian: 2

    let mut x_val = 0.0f32; // Start from x=0
    let target = 3.0f32;

    println!("Minimizing f(x) = (x-3)² using Newton's method");
    println!("Optimal solution: x* = 3.0");
    println!("Starting point: x₀ = {:.4}\n", x_val);

    for iter in 0..5 {
        let tape = GradientTape::new();
        let x = tape.watch(Tensor::from_array(array![x_val].into_dyn()));

        // Compute (x - 3)²
        let target_tensor = tape.watch(Tensor::from_array(array![target].into_dyn()));
        let diff = x.sub(&target_tensor)?;
        let loss = diff.mul(&diff)?;
        let scalar_loss = loss.sum(None, false)?;

        // Get first derivative (gradient)
        let grads = tape.gradient(std::slice::from_ref(&scalar_loss), std::slice::from_ref(&x))?;

        if let Some(grad) = &grads[0] {
            if let Some(grad_data) = grad.as_slice() {
                let gradient = grad_data[0];

                // For f(x) = (x-3)², Hessian is constant = 2
                let hessian = 2.0f32;

                // Newton update: x_new = x - H⁻¹ * g
                let newton_step = gradient / hessian;
                x_val -= newton_step;

                let loss_val = if let Some(data) = scalar_loss.tensor.as_slice() {
                    data[0]
                } else {
                    0.0
                };

                println!(
                    "Iteration {}: x = {:.6}, f(x) = {:.6}, ∇f = {:.6}",
                    iter + 1,
                    x_val,
                    loss_val,
                    gradient
                );
            }
        }
    }

    println!("\nNewton's method converges in 1-2 iterations!");
    println!("Final x ≈ {:.6} (target: 3.0)", x_val);
    println!("Error: {:.6}\n", (x_val - 3.0).abs());

    Ok(())
}
