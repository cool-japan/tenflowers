use tenflowers_autograd::{GradientTape, TrackedTensor};
use tenflowers_core::Tensor;

#[test]
fn test_gradient_tape_basic_add() {
    // Create a gradient tape
    let tape = GradientTape::new();

    // Create input tensors
    let a = Tensor::<f32>::from_vec(vec![2.0, 3.0], &[2]).unwrap();
    let b = Tensor::<f32>::from_vec(vec![4.0, 5.0], &[2]).unwrap();

    // Watch the tensors
    let a_tracked = tape.watch(a);
    let b_tracked = tape.watch(b);

    // Perform addition
    let c = a_tracked.add(&b_tracked).unwrap();

    // Compute gradients
    let gradients = tape.gradient(&[c], &[a_tracked, b_tracked]).unwrap();

    // Check gradients (should be all ones for addition)
    assert_eq!(gradients.len(), 2);

    let grad_a = &gradients[0];
    let grad_b = &gradients[1];

    // Unwrap the Option<Tensor> results
    let grad_a_tensor = grad_a.as_ref().unwrap();
    let grad_b_tensor = grad_b.as_ref().unwrap();

    assert_eq!(grad_a_tensor.shape().dims(), &[2]);
    assert_eq!(grad_b_tensor.shape().dims(), &[2]);

    // For addition, gradients should be 1.0
    let grad_a_data = grad_a_tensor.as_slice().unwrap();
    let grad_b_data = grad_b_tensor.as_slice().unwrap();

    assert!((grad_a_data[0] - 1.0).abs() < 1e-6);
    assert!((grad_a_data[1] - 1.0).abs() < 1e-6);
    assert!((grad_b_data[0] - 1.0).abs() < 1e-6);
    assert!((grad_b_data[1] - 1.0).abs() < 1e-6);
}

#[test]
fn test_gradient_tape_multiplication() {
    let tape = GradientTape::new();

    // Create input tensors
    let a = Tensor::<f32>::from_vec(vec![2.0, 3.0], &[2]).unwrap();
    let b = Tensor::<f32>::from_vec(vec![4.0, 5.0], &[2]).unwrap();

    // Watch the tensors
    let a_tracked = tape.watch(a);
    let b_tracked = tape.watch(b);

    // Perform multiplication
    let c = a_tracked.mul(&b_tracked).unwrap();

    // Compute gradients
    let gradients = tape.gradient(&[c], &[a_tracked, b_tracked]).unwrap();

    let grad_a = &gradients[0];
    let grad_b = &gradients[1];

    // Unwrap the Option<Tensor> results
    let grad_a_tensor = grad_a.as_ref().unwrap();
    let grad_b_tensor = grad_b.as_ref().unwrap();

    // For multiplication: grad_a = b, grad_b = a
    let grad_a_data = grad_a_tensor.as_slice().unwrap();
    let grad_b_data = grad_b_tensor.as_slice().unwrap();

    assert!((grad_a_data[0] - 4.0).abs() < 1e-6); // grad_a[0] = b[0]
    assert!((grad_a_data[1] - 5.0).abs() < 1e-6); // grad_a[1] = b[1]
    assert!((grad_b_data[0] - 2.0).abs() < 1e-6); // grad_b[0] = a[0]
    assert!((grad_b_data[1] - 3.0).abs() < 1e-6); // grad_b[1] = a[1]
}

#[test]
fn test_gradient_tape_chain_rule() {
    let tape = GradientTape::new();

    // Create input tensors
    let x = Tensor::<f32>::from_vec(vec![2.0], &[1]).unwrap();
    let w = Tensor::<f32>::from_vec(vec![3.0], &[1]).unwrap();
    let b = Tensor::<f32>::from_vec(vec![1.0], &[1]).unwrap();

    // Watch the tensors
    let x_tracked = tape.watch(x);
    let w_tracked = tape.watch(w);
    let b_tracked = tape.watch(b);

    // Compute y = w * x + b
    let wx = w_tracked.mul(&x_tracked).unwrap();
    let y = wx.add(&b_tracked).unwrap();

    // Compute gradients
    let gradients = tape
        .gradient(&[y], &[x_tracked, w_tracked, b_tracked])
        .unwrap();

    let grad_x = &gradients[0];
    let grad_w = &gradients[1];
    let grad_b = &gradients[2];

    // Unwrap the Option<Tensor> results
    let grad_x_tensor = grad_x.as_ref().unwrap();
    let grad_w_tensor = grad_w.as_ref().unwrap();
    let grad_b_tensor = grad_b.as_ref().unwrap();

    // Check gradients
    let grad_x_data = grad_x_tensor.as_slice().unwrap();
    let grad_w_data = grad_w_tensor.as_slice().unwrap();
    let grad_b_data = grad_b_tensor.as_slice().unwrap();

    assert!((grad_x_data[0] - 3.0).abs() < 1e-6); // dy/dx = w = 3.0
    assert!((grad_w_data[0] - 2.0).abs() < 1e-6); // dy/dw = x = 2.0
    assert!((grad_b_data[0] - 1.0).abs() < 1e-6); // dy/db = 1.0
}

#[test]
fn test_gradient_tape_relu() {
    let tape = GradientTape::new();

    // Create input tensor with positive and negative values
    let x = Tensor::<f32>::from_vec(vec![2.0, -1.0, 3.0, -2.0], &[4]).unwrap();

    // Watch the tensor
    let x_tracked = tape.watch(x);

    // Apply ReLU
    let y = x_tracked.relu().unwrap();

    // Compute gradients
    let gradients = tape.gradient(&[y], &[x_tracked]).unwrap();

    let grad_x = &gradients[0];
    let grad_x_tensor = grad_x.as_ref().unwrap();
    let grad_x_data = grad_x_tensor.as_slice().unwrap();

    // Check gradients: should be 1 for positive inputs, 0 for negative
    assert!((grad_x_data[0] - 1.0).abs() < 1e-6); // x[0] > 0
    assert!((grad_x_data[1] - 0.0).abs() < 1e-6); // x[1] < 0
    assert!((grad_x_data[2] - 1.0).abs() < 1e-6); // x[2] > 0
    assert!((grad_x_data[3] - 0.0).abs() < 1e-6); // x[3] < 0
}

#[test]
fn test_gradient_tape_matmul() {
    let tape = GradientTape::new();

    // Create 2x3 and 3x2 matrices
    let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let b = Tensor::<f32>::from_vec(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]).unwrap();

    // Watch the tensors
    let a_tracked = tape.watch(a);
    let b_tracked = tape.watch(b);

    // Perform matrix multiplication
    let c = a_tracked.matmul(&b_tracked).unwrap();

    // The result should be 2x2
    assert_eq!(c.tensor().shape().dims(), &[2, 2]);

    // Compute gradients
    let gradients = tape.gradient(&[c], &[a_tracked, b_tracked]).unwrap();

    let grad_a = &gradients[0];
    let grad_b = &gradients[1];

    // Unwrap the Option<Tensor> results
    let grad_a_tensor = grad_a.as_ref().unwrap();
    let grad_b_tensor = grad_b.as_ref().unwrap();

    // Check shapes
    assert_eq!(grad_a_tensor.shape().dims(), &[2, 3]);
    assert_eq!(grad_b_tensor.shape().dims(), &[3, 2]);

    // For matmul with identity gradient:
    // grad_a = grad_output @ b.T
    // grad_b = a.T @ grad_output
    // Since grad_output is identity (all ones), we can verify the sums
    let grad_a_data = grad_a_tensor.as_slice().unwrap();
    let grad_b_data = grad_b_tensor.as_slice().unwrap();

    // grad_a should be [[15, 19, 23], [15, 19, 23]] (sum of b's columns)
    assert!((grad_a_data[0] - 15.0).abs() < 1e-6);
    assert!((grad_a_data[1] - 19.0).abs() < 1e-6);
    assert!((grad_a_data[2] - 23.0).abs() < 1e-6);
    assert!((grad_a_data[3] - 15.0).abs() < 1e-6);
    assert!((grad_a_data[4] - 19.0).abs() < 1e-6);
    assert!((grad_a_data[5] - 23.0).abs() < 1e-6);

    // grad_b should be [[5, 5], [7, 7], [9, 9]] (sum of a's rows)
    assert!((grad_b_data[0] - 5.0).abs() < 1e-6);
    assert!((grad_b_data[1] - 5.0).abs() < 1e-6);
    assert!((grad_b_data[2] - 7.0).abs() < 1e-6);
    assert!((grad_b_data[3] - 7.0).abs() < 1e-6);
    assert!((grad_b_data[4] - 9.0).abs() < 1e-6);
    assert!((grad_b_data[5] - 9.0).abs() < 1e-6);
}

#[test]
fn test_gradient_tape_complex_computation() {
    let tape = GradientTape::new();

    // Create input tensors
    let x = Tensor::<f32>::from_vec(vec![1.0, 2.0], &[2]).unwrap();
    let w1 = Tensor::<f32>::from_vec(vec![0.5, -0.5], &[2]).unwrap();
    let w2 = Tensor::<f32>::from_vec(vec![1.0, 1.0], &[2]).unwrap();

    // Watch the tensors
    let x_tracked = tape.watch(x);
    let w1_tracked = tape.watch(w1);
    let w2_tracked = tape.watch(w2);

    // Compute y = relu(w1 * x) * w2
    let w1x = w1_tracked.mul(&x_tracked).unwrap();
    let relu_out = w1x.relu().unwrap();
    let y = relu_out.mul(&w2_tracked).unwrap();

    // Compute gradients
    let gradients = tape
        .gradient(&[y], &[x_tracked, w1_tracked, w2_tracked])
        .unwrap();

    let grad_x = &gradients[0];
    let grad_w1 = &gradients[1];
    let grad_w2 = &gradients[2];

    // Unwrap the Option<Tensor> results
    let grad_x_tensor = grad_x.as_ref().unwrap();
    let grad_w1_tensor = grad_w1.as_ref().unwrap();
    let grad_w2_tensor = grad_w2.as_ref().unwrap();

    // Verify gradient shapes
    assert_eq!(grad_x_tensor.shape().dims(), &[2]);
    assert_eq!(grad_w1_tensor.shape().dims(), &[2]);
    assert_eq!(grad_w2_tensor.shape().dims(), &[2]);

    // w1 * x = [0.5, -1.0]
    // relu(w1 * x) = [0.5, 0.0]
    // y = [0.5, 0.0]

    // grad_w2 = relu(w1 * x) = [0.5, 0.0]
    let grad_w2_data = grad_w2_tensor.as_slice().unwrap();
    assert!((grad_w2_data[0] - 0.5).abs() < 1e-6);
    assert!((grad_w2_data[1] - 0.0).abs() < 1e-6);

    // For grad_x and grad_w1, need to propagate through relu
    // grad_relu = w2 * (w1*x > 0) = [1.0, 0.0]
    // grad_x = w1 * grad_relu = [0.5, 0.0]
    let grad_x_data = grad_x_tensor.as_slice().unwrap();
    assert!((grad_x_data[0] - 0.5).abs() < 1e-6);
    assert!((grad_x_data[1] - 0.0).abs() < 1e-6);

    // grad_w1 = x * grad_relu = [1.0, 0.0]
    let grad_w1_data = grad_w1_tensor.as_slice().unwrap();
    assert!((grad_w1_data[0] - 1.0).abs() < 1e-6);
    assert!((grad_w1_data[1] - 0.0).abs() < 1e-6);
}

#[test]
fn test_gradient_tape_broadcasting() {
    let tape = GradientTape::new();

    // Create tensors with different shapes for broadcasting
    let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], &[3, 1]).unwrap();
    let b = Tensor::<f32>::from_vec(vec![4.0, 5.0], &[1, 2]).unwrap();

    // Watch the tensors
    let a_tracked = tape.watch(a);
    let b_tracked = tape.watch(b);

    // Perform addition with broadcasting
    let c = a_tracked.add(&b_tracked).unwrap();

    // The result should be 3x2
    assert_eq!(c.tensor().shape().dims(), &[3, 2]);

    // Compute gradients
    let gradients = tape.gradient(&[c], &[a_tracked, b_tracked]).unwrap();

    let grad_a = &gradients[0];
    let grad_b = &gradients[1];

    // Check shapes - gradients should have original shapes
    assert_eq!(grad_a.as_ref().unwrap().shape().dims(), &[3, 1]);
    assert_eq!(grad_b.as_ref().unwrap().shape().dims(), &[1, 2]);

    // grad_a should sum over the broadcasted dimension
    let grad_a_data = grad_a.as_ref().unwrap().as_slice().unwrap();
    assert!((grad_a_data[0] - 2.0).abs() < 1e-6); // sum of 1s across dim 1
    assert!((grad_a_data[1] - 2.0).abs() < 1e-6);
    assert!((grad_a_data[2] - 2.0).abs() < 1e-6);

    // grad_b should sum over the broadcasted dimension
    let grad_b_data = grad_b.as_ref().unwrap().as_slice().unwrap();
    assert!((grad_b_data[0] - 3.0).abs() < 1e-6); // sum of 1s across dim 0
    assert!((grad_b_data[1] - 3.0).abs() < 1e-6);
}
