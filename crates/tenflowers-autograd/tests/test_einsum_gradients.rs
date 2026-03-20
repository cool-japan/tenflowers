use tenflowers_autograd::{GradientTape, TrackedTensor};
use tenflowers_core::Tensor;

#[test]
fn test_einsum_matrix_multiply_gradient() {
    // Test einsum for matrix multiplication: "ij,jk->ik"
    let tape = GradientTape::new();

    // Create test matrices
    let a_data = vec![1.0, 2.0, 3.0, 4.0]; // 2x2 matrix
    let b_data = vec![5.0, 6.0, 7.0, 8.0]; // 2x2 matrix

    let a_tensor = Tensor::from_vec(a_data, &[2, 2]).unwrap();
    let b_tensor = Tensor::from_vec(b_data, &[2, 2]).unwrap();

    let a_tracked = tape.watch(a_tensor);
    let b_tracked = tape.watch(b_tensor);

    // Perform einsum: C = A @ B
    let c_tracked = TrackedTensor::einsum("ij,jk->ik", &[&a_tracked, &b_tracked]).unwrap();

    // Compute gradients
    let gradients = tape
        .gradient(&[c_tracked], &[a_tracked, b_tracked])
        .unwrap();

    // Debug: Print gradient status
    println!("Gradient 0 (A): {:?}", gradients[0].is_some());
    println!("Gradient 1 (B): {:?}", gradients[1].is_some());

    // Verify gradients have correct shapes
    let grad_a = gradients[0]
        .as_ref()
        .expect("Gradient for A should not be None");
    let grad_b = gradients[1]
        .as_ref()
        .expect("Gradient for B should not be None");
    assert_eq!(grad_a.shape().dims(), &[2, 2]); // gradient w.r.t. A
    assert_eq!(grad_b.shape().dims(), &[2, 2]); // gradient w.r.t. B

    println!("Einsum matrix multiplication gradient test passed!");
}

#[test]
fn test_einsum_transpose_gradient() {
    // Test einsum for transpose: "ij->ji"
    let tape = GradientTape::new();

    let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3 matrix
    let a_tensor = Tensor::from_vec(a_data, &[2, 3]).unwrap();
    let a_tracked = tape.watch(a_tensor);

    // Perform einsum transpose
    let result = TrackedTensor::einsum("ij->ji", &[&a_tracked]).unwrap();

    // Compute gradient
    let gradients = tape.gradient(&[result], &[a_tracked]).unwrap();

    // Verify gradient has correct shape (same as original)
    let grad_tensor = gradients[0].as_ref().unwrap();
    assert_eq!(grad_tensor.shape().dims(), &[2, 3]);

    println!("Einsum transpose gradient test passed!");
}

#[test]
fn test_einsum_elementwise_gradient() {
    // Test einsum for element-wise multiplication: "ij,ij->ij"
    let tape = GradientTape::new();

    let a_data = vec![1.0, 2.0, 3.0, 4.0];
    let b_data = vec![2.0, 3.0, 4.0, 5.0];

    let a_tensor = Tensor::from_vec(a_data, &[2, 2]).unwrap();
    let b_tensor = Tensor::from_vec(b_data, &[2, 2]).unwrap();

    let a_tracked = tape.watch(a_tensor);
    let b_tracked = tape.watch(b_tensor);

    // Element-wise multiplication via einsum
    let result = TrackedTensor::einsum("ij,ij->ij", &[&a_tracked, &b_tracked]).unwrap();

    // Compute gradients
    let gradients = tape.gradient(&[result], &[a_tracked, b_tracked]).unwrap();

    // Verify shapes
    let grad_a = gradients[0].as_ref().unwrap();
    let grad_b = gradients[1].as_ref().unwrap();
    assert_eq!(grad_a.shape().dims(), &[2, 2]);
    assert_eq!(grad_b.shape().dims(), &[2, 2]);

    println!("Einsum element-wise gradient test passed!");
}
