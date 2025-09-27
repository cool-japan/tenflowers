use tenflowers_autograd::{GradientTape, TrackedTensor};
use tenflowers_core::Tensor;

#[test]
fn test_hessian_simple_quadratic() {
    let tape = GradientTape::new();

    // f(x) = x^2, where x is a scalar
    // Hessian should be [2]
    let x = Tensor::<f32>::from_vec(vec![3.0], &[1]).unwrap();
    let x_tracked = tape.watch(x);

    let x_squared = x_tracked.mul(&x_tracked).unwrap();

    // Test the Hessian computation for f(x) = x^2
    // The second derivative should be 2
    let result = tape.hessian(&x_squared, &x_tracked);
    assert!(result.is_ok(), "Hessian computation should succeed");

    let hessian = result.unwrap();
    assert_eq!(
        hessian.shape().dims(),
        &[1, 1],
        "Hessian should be 1x1 matrix"
    );

    let hessian_data = hessian.as_slice().unwrap();
    // For f(x) = x^2, the Hessian should be [2]
    assert!(
        (hessian_data[0] - 2.0).abs() < 1e-6,
        "Hessian should be approximately 2.0, got {}",
        hessian_data[0]
    );

    println!("✅ Hessian test for x^2 passed: H = {}", hessian_data[0]);
}

#[test]
fn test_hessian_multivariate_function() {
    let tape = GradientTape::new();

    // f(x, y) = x^2 + y^2, where x and y are elements of a 2D vector
    // Hessian should be [[2, 0], [0, 2]]
    let xy = Tensor::<f32>::from_vec(vec![1.0, 2.0], &[2]).unwrap();
    let xy_tracked = tape.watch(xy);

    let xy_squared = xy_tracked.mul(&xy_tracked).unwrap();
    let sum_xy_squared = xy_squared.sum(Some(vec![0]), false).unwrap();

    // Test the Hessian computation for f(x,y) = x^2 + y^2
    // The Hessian should be [[2, 0], [0, 2]]
    let result = tape.hessian(&sum_xy_squared, &xy_tracked);
    assert!(
        result.is_ok(),
        "Multivariate Hessian computation should succeed"
    );

    let hessian = result.unwrap();
    assert_eq!(
        hessian.shape().dims(),
        &[2, 2],
        "Hessian should be 2x2 matrix"
    );

    let hessian_data = hessian.as_slice().unwrap();
    // For f(x,y) = x^2 + y^2, the Hessian should be diag([2, 2])
    assert!(
        (hessian_data[0] - 2.0).abs() < 1e-6,
        "H[0,0] should be 2.0, got {}",
        hessian_data[0]
    ); // ∂²f/∂x²
    assert!(
        hessian_data[1].abs() < 1e-6,
        "H[0,1] should be 0.0, got {}",
        hessian_data[1]
    ); // ∂²f/∂x∂y
    assert!(
        hessian_data[2].abs() < 1e-6,
        "H[1,0] should be 0.0, got {}",
        hessian_data[2]
    ); // ∂²f/∂y∂x
    assert!(
        (hessian_data[3] - 2.0).abs() < 1e-6,
        "H[1,1] should be 2.0, got {}",
        hessian_data[3]
    ); // ∂²f/∂y²

    println!(
        "✅ Multivariate Hessian test passed: H = [[{}, {}], [{}, {}]]",
        hessian_data[0], hessian_data[1], hessian_data[2], hessian_data[3]
    );
}

#[test]
fn test_jacobian_vector_product_api() {
    let tape = GradientTape::new();

    // Test JVP API
    let x = Tensor::<f32>::from_vec(vec![1.0, 2.0], &[2]).unwrap();
    let x_tracked = tape.watch(x);

    let y = x_tracked.mul(&x_tracked).unwrap();

    let v = vec![Tensor::<f32>::from_vec(vec![1.0, 0.0], &[2]).unwrap()];

    // Test JVP computation
    let result = tape.jvp(&[&y], &[&x_tracked], &v);

    // Test that API works
    assert!(result.is_ok());
    let jvp_result = result.unwrap();
    assert_eq!(jvp_result.len(), 1);
    println!("JVP API test completed successfully");
}

#[test]
fn test_jvp_comprehensive() {
    // Test JVP on a simple quadratic function f(x,y) = x^2 + 2*x*y + y^2
    let tape = GradientTape::new();

    let x = Tensor::<f32>::from_vec(vec![1.0], &[1]).unwrap();
    let y = Tensor::<f32>::from_vec(vec![2.0], &[1]).unwrap();

    let x_tracked = tape.watch(x);
    let y_tracked = tape.watch(y);

    // f = x^2 + 2*x*y + y^2
    let x_squared = x_tracked.mul(&x_tracked).unwrap();
    let y_squared = y_tracked.mul(&y_tracked).unwrap();
    let xy = x_tracked.mul(&y_tracked).unwrap();
    let two_xy = xy.add(&xy).unwrap(); // 2*x*y
    let f = x_squared.add(&two_xy).unwrap().add(&y_squared).unwrap();

    // Test JVP with direction vector v = [1, 1]
    let v1 = Tensor::<f32>::from_vec(vec![1.0], &[1]).unwrap();
    let v2 = Tensor::<f32>::from_vec(vec![1.0], &[1]).unwrap();

    let result = tape.jvp(&[&f], &[&x_tracked, &y_tracked], &[v1, v2]);
    assert!(result.is_ok());

    let jvp_values = result.unwrap();
    assert_eq!(jvp_values.len(), 1);

    // For f(x,y) = x^2 + 2*x*y + y^2 at (1, 2) with direction (1, 1):
    // ∇f = [2*x + 2*y, 2*x + 2*y] = [2*1 + 2*2, 2*1 + 2*2] = [6, 6]
    // JVP = ∇f · v = [6, 6] · [1, 1] = 12

    let jvp_scalar = jvp_values[0].as_slice().unwrap()[0];
    assert!(
        (jvp_scalar - 12.0).abs() < 1e-6,
        "JVP should be 12.0, got {}",
        jvp_scalar
    );

    println!("✅ Comprehensive JVP test passed: JVP = {}", jvp_scalar);
}

#[test]
fn test_numerical_gradient_checking() {
    let tape = GradientTape::new();

    // Test function: f(x) = x^2 + 2*x
    // Analytical gradient: f'(x) = 2*x + 2
    let test_function =
        |inputs: &[&TrackedTensor<f32>]| -> tenflowers_core::Result<TrackedTensor<f32>> {
            let x = inputs[0];
            let x_squared = x.mul(x)?;
            let two_x = x.add(x)?;
            let result = x_squared.add(&two_x)?;

            // Return sum for scalar output - sum all elements to get a scalar
            result.sum(None, false)
        };

    let x = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let epsilon = 1e-5;
    let relative_tolerance = 2e-2; // More lenient for finite differences
    let absolute_tolerance = 2e-2; // More lenient for finite differences

    // Test numerical gradient checking
    let result = GradientTape::numerical_gradient_check(
        test_function,
        &[x],
        epsilon,
        relative_tolerance,
        absolute_tolerance,
    );

    // The test should work even though we don't have full computation graph replay
    // This mainly tests that the API works
    println!("Numerical gradient check result: {:?}", result);
    assert!(result.is_ok());
}

#[test]
fn test_numerical_gradient_simple_quadratic() {
    let tape = GradientTape::new();

    // Simple test: f(x) = x^2 where x is scalar
    // Gradient should be 2*x
    let quadratic_function =
        |inputs: &[&TrackedTensor<f32>]| -> tenflowers_core::Result<TrackedTensor<f32>> {
            let x = inputs[0];
            x.mul(x)
        };

    let x = Tensor::<f32>::from_vec(vec![3.0], &[1]).unwrap();

    // Test numerical gradient checking for simple quadratic
    // Use more reasonable tolerances for finite differences
    let result = GradientTape::numerical_gradient_check(quadratic_function, &[x], 1e-5, 1e-2, 1e-2);

    match result {
        Ok(()) => println!("Simple quadratic gradient check passed"),
        Err(e) => {
            println!("Gradient check failed with error: {:?}", e);
            panic!("Test failed: {:?}", e);
        }
    }
}
