use scirs2_core::ndarray::Array1;
/// Example demonstrating gradient validation concept
///
/// To run the actual gradient correctness tests, use:
/// cargo test gradient_correctness
///
/// Requirements for full validation:
/// - Python 3 with PyTorch and NumPy installed
/// - pip install torch numpy
use tenflowers_autograd::GradientTape;
use tenflowers_core::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("TenfloweRS Gradient Validation Example");
    println!("=====================================\n");

    println!("This example shows how TenfloweRS computes gradients.");
    println!("For full validation against PyTorch, run: cargo test gradient_correctness\n");

    // Simple gradient computation example
    let tape = GradientTape::new();

    // Create input tensors
    let x_data = Array1::linspace(0.5f32, 2.0, 5).into_dyn();
    let y_data = Array1::linspace(1.0f32, 3.0, 5).into_dyn();

    let x = tape.watch(Tensor::from_array(x_data));
    let y = tape.watch(Tensor::from_array(y_data));

    // Perform operations
    let z = x.mul(&y)?;
    let loss = z.sum(None, false)?;

    // Compute gradients
    let grads = tape.gradient(&[loss.clone()], &[x.clone(), y.clone()])?;

    println!("Input x: {:?}", x.tensor());
    println!("Input y: {:?}", y.tensor());
    println!("Result z = x * y: {:?}", z.tensor());
    println!("Loss = sum(z): {:?}", loss.tensor());
    println!("Gradient dx: {:?}", &grads[0]);
    println!("Gradient dy: {:?}", &grads[1]);

    println!("\n✓ Gradient computation successful!");
    println!("\n📝 Note: The gradient of x should equal y, and gradient of y should equal x");
    println!("   This is because d/dx(x*y) = y and d/dy(x*y) = x");

    println!("\n🔍 To run comprehensive gradient validation against PyTorch:");
    println!("   cargo test gradient_correctness_basic");
    println!("   cargo test gradient_correctness_activations");
    println!("   cargo test test_full_gradient_correctness_suite -- --ignored");

    Ok(())
}
