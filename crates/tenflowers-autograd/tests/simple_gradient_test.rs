use tenflowers_autograd::GradientTape;
use tenflowers_core::Tensor;

#[test]
fn test_simple_gradient() {
    // Create a gradient tape
    let tape = GradientTape::new();

    // Create simple scalars
    let mut x = Tensor::<f32>::from_scalar(2.0);
    let mut y = Tensor::<f32>::from_scalar(3.0);

    // Enable gradient tracking
    x.set_requires_grad(true);
    y.set_requires_grad(true);

    // Watch the tensors
    let x_tracked = tape.watch(x);
    let y_tracked = tape.watch(y);

    // Compute z = x + y
    let z = x_tracked.add(&y_tracked).unwrap();

    // Compute gradients
    let gradients = tape.gradient(&[z], &[x_tracked, y_tracked]).unwrap();

    assert_eq!(gradients.len(), 2);

    // Both gradients should be 1.0 for addition
    let grad_x = gradients[0].as_ref().unwrap().get(&[]).unwrap();
    let grad_y = gradients[1].as_ref().unwrap().get(&[]).unwrap();

    assert!((grad_x - 1.0).abs() < 1e-6);
    assert!((grad_y - 1.0).abs() < 1e-6);
}

#[test]
fn test_pow_operation() {
    // Create a gradient tape
    let tape = GradientTape::new();

    // Test TrackedTensor pow operation: x^y where x=2, y=3
    let x = Tensor::<f32>::from_scalar(2.0);
    let y = Tensor::<f32>::from_scalar(3.0);

    let x_tracked = tape.watch(x);
    let y_tracked = tape.watch(y);

    // Compute z = x^y = 2^3 = 8
    let z = x_tracked.pow(&y_tracked).unwrap();

    // Verify the result
    let result_value = z.tensor.get(&[]).unwrap();
    assert!(
        (result_value - 8.0).abs() < 1e-6,
        "Expected 2^3 = 8.0, got {}",
        result_value
    );

    // Test gradient computation
    let gradients = tape.gradient(&[z], &[x_tracked, y_tracked]).unwrap();

    // For z = x^y:
    // dz/dx = y * x^(y-1) = 3 * 2^2 = 12
    // dz/dy = x^y * ln(x) = 8 * ln(2) ≈ 5.545

    let grad_x = gradients[0].as_ref().unwrap().get(&[]).unwrap();
    let grad_y = gradients[1].as_ref().unwrap().get(&[]).unwrap();

    assert!(
        (grad_x - 12.0).abs() < 1e-5,
        "Expected dz/dx = 12.0, got {}",
        grad_x
    );
    assert!(
        (grad_y - (8.0 * 2.0_f32.ln())).abs() < 0.3,
        "Expected dz/dy ≈ {}, got {}",
        8.0 * 2.0_f32.ln(),
        grad_y
    );
}

#[test]
fn test_softmax_operation() {
    // Create a gradient tape
    let tape = GradientTape::new();

    // Test TrackedTensor softmax operation
    // Input logits: [1.0, 2.0, 3.0]
    let x = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let x_tracked = tape.watch(x);

    // Compute softmax along the last axis
    let y = x_tracked.softmax(Some(-1)).unwrap();

    // Verify the result - softmax should sum to 1.0
    let result_data = y.tensor.as_slice().unwrap();
    let sum: f32 = result_data.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-6,
        "Softmax should sum to 1.0, got {}",
        sum
    );

    // Verify the softmax values are reasonable (increasing exponentially)
    assert!(result_data[0] < result_data[1]);
    assert!(result_data[1] < result_data[2]);

    // Test gradient computation
    let gradients = tape
        .gradient(std::slice::from_ref(&y), &[x_tracked])
        .unwrap();

    let grad_x = gradients[0].as_ref().unwrap();

    // Gradient should have same shape as input
    assert_eq!(grad_x.shape().dims(), &[3]);

    // For a single softmax output with gradient 1.0, the gradient should satisfy:
    // sum of gradients should be 0 (since softmax outputs sum to 1, gradients must sum to 0)
    let grad_data = grad_x.as_slice().unwrap();
    let grad_sum: f32 = grad_data.iter().sum();
    assert!(
        grad_sum.abs() < 1e-5,
        "Softmax gradients should sum to ~0, got {}",
        grad_sum
    );

    // Manual verification: compute expected softmax values
    let exp_vals: Vec<f32> = vec![1.0_f32.exp(), 2.0_f32.exp(), 3.0_f32.exp()];
    let exp_sum: f32 = exp_vals.iter().sum();
    let expected_softmax: Vec<f32> = exp_vals.iter().map(|&x| x / exp_sum).collect();

    for (i, &expected) in expected_softmax.iter().enumerate() {
        assert!(
            (result_data[i] - expected).abs() < 1e-6,
            "Softmax[{}] expected {}, got {}",
            i,
            expected,
            result_data[i]
        );
    }
}
