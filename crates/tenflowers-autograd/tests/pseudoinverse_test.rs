use tenflowers_autograd::{GradientTape, TrackedTensor};
use tenflowers_core::Tensor;

#[test]
fn test_pseudoinverse_forward() {
    let tape = GradientTape::new();

    // Create a simple 2x3 matrix for testing pseudoinverse
    let data = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32];
    let matrix = Tensor::from_vec(data, &[2, 3]).unwrap();
    let tracked_matrix = tape.watch(matrix);

    // Compute pseudoinverse
    let pinv_result = tracked_matrix.pinv();
    if let Err(e) = &pinv_result {
        println!("Pseudoinverse error: {:?}", e);
    }
    assert!(pinv_result.is_ok());

    let pinv_tensor = pinv_result.unwrap();

    // Check that pseudoinverse has correct shape (3x2 for a 2x3 input)
    assert_eq!(pinv_tensor.tensor.shape().dims(), &[3, 2]);
}

#[test]
fn test_pseudoinverse_gradient() {
    let tape = GradientTape::new();

    // Create a simple square matrix for testing gradients
    let data = vec![2.0f32, 1.0f32, 1.0f32, 2.0f32];
    let matrix = Tensor::from_vec(data, &[2, 2]).unwrap();
    let tracked_matrix = tape.watch(matrix);

    // Compute pseudoinverse
    let pinv_result = tracked_matrix.pinv().unwrap();

    // Create a simple loss (sum of all elements in pseudoinverse)
    let loss = pinv_result.sum(None, false).unwrap();

    // Compute gradients
    let gradients = tape.gradient(&[loss], &[tracked_matrix]);
    if let Err(ref e) = gradients {
        eprintln!("Gradient computation error: {:?}", e);
    }
    assert!(
        gradients.is_ok(),
        "Gradient computation failed: {:?}",
        gradients.err()
    );

    let grad_result = gradients.unwrap();
    assert_eq!(grad_result.len(), 1);

    // Check that gradient has same shape as input
    println!("Gradient result: {:?}", grad_result[0].is_some());
    if grad_result[0].is_none() {
        println!("ERROR: Gradient is None!");
        panic!("Gradient computation failed - got None instead of gradient tensor");
    }
    let grad_tensor = grad_result[0].as_ref().unwrap();
    println!("Gradient shape: {:?}", grad_tensor.shape().dims());
    assert_eq!(grad_tensor.shape().dims(), &[2, 2]);
}

#[test]
fn test_pseudoinverse_rectangular_matrix() {
    let tape = GradientTape::new();

    // Create a tall rectangular matrix (3x2)
    let data = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32];
    let matrix = Tensor::from_vec(data, &[3, 2]).unwrap();
    let tracked_matrix = tape.watch(matrix);

    // Compute pseudoinverse
    let pinv_result = tracked_matrix.pinv();
    if let Err(e) = &pinv_result {
        println!("Rectangular matrix pseudoinverse error: {:?}", e);
    }
    assert!(pinv_result.is_ok());

    let pinv_tensor = pinv_result.unwrap();

    // Check that pseudoinverse has correct shape (2x3 for a 3x2 input)
    assert_eq!(pinv_tensor.tensor.shape().dims(), &[2, 3]);
}

#[test]
fn test_pseudoinverse_properties() {
    // Test mathematical properties of pseudoinverse
    let tape = GradientTape::new();

    // Create a simple matrix
    let data = vec![1.0f32, 0.0f32, 0.0f32, 1.0f32];
    let matrix = Tensor::from_vec(data, &[2, 2]).unwrap();
    let tracked_matrix = tape.watch(matrix);

    // For an identity matrix, pseudoinverse should be itself
    let pinv_result = tracked_matrix.pinv().unwrap();

    // A * A^+ should be approximately A for a full-rank matrix
    let a_pinv_a = tracked_matrix.matmul(&pinv_result).unwrap();

    // The result should be close to the identity matrix
    let identity_diff = a_pinv_a.tensor.sub(&Tensor::eye(2)).unwrap();

    // Check that the difference is small (numerical tolerance)
    if let Some(diff_data) = identity_diff.as_slice() {
        for &val in diff_data {
            assert!(
                val.abs() < 1e-6,
                "Pseudoinverse property A*A^+ ≈ A failed: diff = {}",
                val
            );
        }
    }
}

#[test]
fn test_pseudoinverse_singular_matrix() {
    let tape = GradientTape::new();

    // Create a singular matrix (rank deficient)
    let data = vec![1.0f32, 2.0f32, 2.0f32, 4.0f32]; // Second row is 2x first row
    let matrix = Tensor::from_vec(data, &[2, 2]).unwrap();
    let tracked_matrix = tape.watch(matrix);

    // Pseudoinverse should still work for singular matrices
    let pinv_result = tracked_matrix.pinv();
    assert!(pinv_result.is_ok());

    let pinv_tensor = pinv_result.unwrap();
    assert_eq!(pinv_tensor.tensor.shape().dims(), &[2, 2]);

    // Test gradient computation works for singular case
    let loss = pinv_tensor.sum(None, false).unwrap();
    let gradients = tape.gradient(&[loss], &[tracked_matrix]);
    if let Err(ref e) = gradients {
        eprintln!("Gradient computation error for singular matrix: {:?}", e);
    }
    assert!(
        gradients.is_ok(),
        "Gradient computation failed for singular matrix: {:?}",
        gradients.err()
    );
}
