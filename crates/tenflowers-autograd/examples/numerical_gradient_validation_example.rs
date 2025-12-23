// # Numerical Gradient Validation Example
//
// This example demonstrates comprehensive numerical gradient validation using
// property-based testing and multiple finite difference methods.
//
// Topics covered:
// - Basic numerical gradient checking
// - Multiple finite difference methods (forward, backward, central, 4-point, 6-point)
// - Property-based testing with random inputs
// - Custom operation gradient validation
// - Adaptive epsilon selection
// - Error analysis and diagnostics

use scirs2_core::ndarray::array;
use tenflowers_autograd::numerical_checker::{
    CheckerConfig, FiniteDifferenceMethod, NumericalChecker,
};
use tenflowers_autograd::GradientTape;
use tenflowers_core::{Result, Tensor};

fn main() -> Result<()> {
    println!("TenfloweRS Numerical Gradient Validation Example");
    println!("================================================\n");

    // Example 1: Basic gradient checking with default settings
    example_1_basic_checking()?;

    // Example 2: Comparing different finite difference methods
    example_2_compare_methods()?;

    // Example 3: Property-based testing
    example_3_property_testing()?;

    // Example 4: Custom operation validation
    example_4_custom_operation()?;

    // Example 5: Adaptive epsilon selection
    example_5_adaptive_epsilon()?;

    // Example 6: Error analysis and diagnostics
    example_6_error_analysis()?;

    println!("\n\nAll gradient validation examples completed successfully!");
    Ok(())
}

/// Example 1: Basic numerical gradient checking
fn example_1_basic_checking() -> Result<()> {
    println!("Example 1: Basic Numerical Gradient Checking");
    println!("--------------------------------------------");

    // Create default checker with central difference
    let config = CheckerConfig::default();
    let mut checker = NumericalChecker::new(config);

    // Create input tensor
    let x = Tensor::from_array(array![2.0f32, 3.0, 4.0].into_dyn());

    // Define function: f(x) = x^2
    let f =
        |tensor: &Tensor<f32>| -> Result<Tensor<f32>> { tenflowers_core::ops::mul(tensor, tensor) };

    // Compute numerical gradient
    let numerical_grad = checker.compute_numerical_gradient(&x, f, 1e-6)?;

    println!("Input: {:?}", x.as_slice().unwrap());
    println!("Function: f(x) = x²");
    println!("Numerical gradient (should be ≈ 2x):");
    println!("  {:?}", numerical_grad.as_slice().unwrap());
    println!("Expected gradient: [4.0, 6.0, 8.0]\n");

    Ok(())
}

/// Example 2: Comparing different finite difference methods
fn example_2_compare_methods() -> Result<()> {
    println!("Example 2: Comparing Finite Difference Methods");
    println!("-----------------------------------------------");

    let x = Tensor::from_array(array![1.0f32, 2.0, 3.0].into_dyn());

    // Function: f(x) = x^3
    let f = |tensor: &Tensor<f32>| -> Result<Tensor<f32>> {
        let x2 = tenflowers_core::ops::mul(tensor, tensor)?;
        tenflowers_core::ops::mul(&x2, tensor)
    };

    let methods = vec![
        ("Forward", FiniteDifferenceMethod::Forward),
        ("Backward", FiniteDifferenceMethod::Backward),
        ("Central", FiniteDifferenceMethod::Central),
        ("Central 4-point", FiniteDifferenceMethod::Central4Point),
        ("Central 6-point", FiniteDifferenceMethod::Central6Point),
    ];

    println!("Input: {:?}", x.as_slice().unwrap());
    println!("Function: f(x) = x³");
    println!("Expected gradient (3x²): [3.0, 12.0, 27.0]\n");

    for (name, method) in methods {
        let mut config = CheckerConfig::default();
        config.method = method;
        let mut checker = NumericalChecker::new(config);

        let epsilon = method.recommended_epsilon();
        let numerical_grad = checker.compute_numerical_gradient(&x, &f, epsilon)?;

        println!(
            "{:20} (h={:.0e}): {:?}",
            name,
            epsilon,
            numerical_grad.as_slice().unwrap()
        );
    }

    println!();
    Ok(())
}

/// Example 3: Property-based testing at multiple random points
fn example_3_property_testing() -> Result<()> {
    println!("Example 3: Property-Based Gradient Testing");
    println!("------------------------------------------");

    let mut config = CheckerConfig::default();
    config.num_samples = 5;
    config.seed = Some(42); // Reproducible
    let num_samples = config.num_samples;
    let mut checker = NumericalChecker::new(config);

    // Test at multiple random points
    let x = Tensor::from_array(array![1.0f32, 2.0, 3.0].into_dyn());

    // Function: f(x) = sin(x) * x
    let f = |tensor: &Tensor<f32>| -> Result<Tensor<f32>> {
        // Simple approximation for demonstration
        // In practice, you would use actual sin function
        tenflowers_core::ops::mul(tensor, tensor)
    };

    println!("Testing gradient at {} random points", num_samples);
    let results = checker.property_test(&x, f)?;

    for (i, result) in results.iter().enumerate() {
        println!(
            "Test {}: {}",
            i + 1,
            if result.is_valid { "✓" } else { "✗" }
        );
        println!("  Max error: {:.2e}", result.max_error);
        println!("  Mean error: {:.2e}", result.mean_error);
    }

    println!();
    Ok(())
}

/// Example 4: Validating custom operation gradients
fn example_4_custom_operation() -> Result<()> {
    println!("Example 4: Custom Operation Gradient Validation");
    println!("-----------------------------------------------");

    // Validate gradient of a custom operation
    let config = CheckerConfig {
        method: FiniteDifferenceMethod::Central4Point,
        rtol: 1e-4,
        atol: 1e-6,
        ..Default::default()
    };
    let mut checker = NumericalChecker::new(config);

    let x = Tensor::from_array(array![1.0f32, 2.0, 3.0, 4.0].into_dyn());

    // Custom operation: softplus(x) = log(1 + exp(x))
    // Approximation for demonstration: x + 0.5 * x^2
    let custom_op = |tensor: &Tensor<f32>| -> Result<Tensor<f32>> {
        let x2 = tenflowers_core::ops::mul(tensor, tensor)?;
        let scaled_x2 = {
            let shape: Vec<usize> = x2.shape().dims().to_vec();
            let half_array = scirs2_core::ndarray::ArrayD::from_elem(shape, 0.5f32);
            let half = Tensor::from_array(half_array);
            tenflowers_core::ops::mul(&half, &x2)?
        };
        tenflowers_core::ops::add(tensor, &scaled_x2)
    };

    let numerical_grad = checker.compute_numerical_gradient(&x, custom_op, 1e-7)?;

    println!("Custom operation: softplus approximation");
    println!("Input: {:?}", x.as_slice().unwrap());
    println!("Numerical gradient:");
    println!("  {:?}", numerical_grad.as_slice().unwrap());

    // Expected gradient for x + 0.5*x^2 is 1 + x
    println!("Expected gradient (1 + x): [2.0, 3.0, 4.0, 5.0]\n");

    Ok(())
}

/// Example 5: Adaptive epsilon selection
fn example_5_adaptive_epsilon() -> Result<()> {
    println!("Example 5: Adaptive Epsilon Selection");
    println!("-------------------------------------");

    let mut config = CheckerConfig::default();
    config.adaptive_epsilon = true;
    config.min_epsilon = 1e-10;
    config.max_epsilon = 1e-2;

    let epsilon = config.epsilon();
    let mut checker = NumericalChecker::new(config);

    // Test with different scales of input
    let test_cases = vec![
        ("Small scale", array![1e-5f32, 2e-5, 3e-5].into_dyn()),
        ("Medium scale", array![1.0f32, 2.0, 3.0].into_dyn()),
        ("Large scale", array![1e5f32, 2e5, 3e5].into_dyn()),
    ];

    let f =
        |tensor: &Tensor<f32>| -> Result<Tensor<f32>> { tenflowers_core::ops::mul(tensor, tensor) };

    for (name, x_array) in test_cases {
        let x = Tensor::from_array(x_array);
        let numerical_grad = checker.compute_numerical_gradient(&x, &f, epsilon)?;

        println!("{}: ", name);
        println!("  Input: {:?}", x.as_slice().unwrap());
        println!("  Gradient: {:?}", numerical_grad.as_slice().unwrap());
    }

    println!();
    Ok(())
}

/// Example 6: Detailed error analysis
fn example_6_error_analysis() -> Result<()> {
    println!("Example 6: Error Analysis and Diagnostics");
    println!("-----------------------------------------");

    let config = CheckerConfig {
        method: FiniteDifferenceMethod::Central,
        rtol: 1e-3,
        atol: 1e-5,
        ..Default::default()
    };
    let checker = NumericalChecker::new(config);

    // Create "analytical" and "numerical" gradients with known differences
    let analytical = Tensor::from_array(array![1.0f32, 2.0, 3.0, 4.0, 5.0].into_dyn());

    // Simulate numerical gradient with some error
    let numerical_data = vec![1.001f32, 2.002, 2.998, 4.001, 5.003];
    let numerical = Tensor::from_array(
        scirs2_core::ndarray::ArrayD::from_shape_vec(vec![5], numerical_data).unwrap(),
    );

    let result = checker.compare_gradients(&analytical, &numerical)?;

    println!("Gradient Comparison Results:");
    println!("{}", result);

    println!("Error Analysis:");
    println!(
        "  Systematic bias: {}",
        if result.error_analysis.is_systematic {
            "YES"
        } else {
            "NO"
        }
    );
    println!(
        "  Mean signed error: {:.2e}",
        result.error_analysis.mean_signed_error
    );
    println!("  Std deviation: {:.2e}", result.error_analysis.std_error);

    println!("\nWorst Errors:");
    for (i, (idx, error)) in result
        .error_analysis
        .worst_indices
        .iter()
        .zip(result.error_analysis.worst_errors.iter())
        .enumerate()
    {
        println!("  {}: Index {} - Error: {:.2e}", i + 1, idx, error);
    }

    println!("\nError Histogram:");
    for (bucket_start, count) in &result.error_analysis.error_histogram {
        let bar = "█".repeat((count * 50 / result.num_elements).max(1));
        println!("  {:.2e}: {} ({})", bucket_start, bar, count);
    }

    println!();
    Ok(())
}
