use tenflowers_autograd::GradientTape;
use tenflowers_core::Tensor;

#[test]
fn test_autograd_binary_operations_comprehensive() {
    let tape = GradientTape::new();

    // Test data
    let a = Tensor::<f32>::from_vec(vec![3.0, 4.0], &[2]).unwrap();
    let b = Tensor::<f32>::from_vec(vec![2.0, 1.0], &[2]).unwrap();

    let a_tracked = tape.watch(a);
    let b_tracked = tape.watch(b);

    // Test subtraction: c = a - b
    let c = a_tracked.sub(&b_tracked).unwrap();
    let grads_sub = tape.gradient(&[c], &[a_tracked, b_tracked]).unwrap();

    // For c = a - b: dc/da = 1, dc/db = -1
    assert_eq!(grads_sub.len(), 2);
    let grad_a = grads_sub[0].as_ref().unwrap();
    let grad_b = grads_sub[1].as_ref().unwrap();
    assert_eq!(grad_a.as_slice().unwrap(), &[1.0, 1.0]); // grad w.r.t a
    assert_eq!(grad_b.as_slice().unwrap(), &[-1.0, -1.0]); // grad w.r.t b

    // Reset tape for next operation
    let tape = GradientTape::new();
    let a_tracked = tape.watch(Tensor::<f32>::from_vec(vec![6.0, 8.0], &[2]).unwrap());
    let b_tracked = tape.watch(Tensor::<f32>::from_vec(vec![2.0, 4.0], &[2]).unwrap());

    // Test division: d = a / b
    let d = a_tracked.div(&b_tracked).unwrap();
    let grads_div = tape.gradient(&[d], &[a_tracked, b_tracked]).unwrap();

    // For d = a / b: dd/da = 1/b, dd/db = -a/(b^2)
    assert_eq!(grads_div.len(), 2);
    assert_eq!(
        grads_div[0].as_ref().unwrap().as_slice().unwrap(),
        &[0.5, 0.25]
    ); // 1/2, 1/4  (grad w.r.t a)
    assert_eq!(
        grads_div[1].as_ref().unwrap().as_slice().unwrap(),
        &[-1.5, -0.5]
    ); // -6/4, -8/16 (grad w.r.t b)
}

#[test]
fn test_autograd_power_operation() {
    let tape = GradientTape::new();

    let a = Tensor::<f32>::from_vec(vec![2.0, 3.0], &[2]).unwrap();
    let b = Tensor::<f32>::from_vec(vec![3.0, 2.0], &[2]).unwrap();

    let a_tracked = tape.watch(a);
    let b_tracked = tape.watch(b);

    // Test power: c = a^b
    let c = a_tracked.pow(&b_tracked).unwrap();
    let grads_pow = tape.gradient(&[c], &[a_tracked, b_tracked]).unwrap();

    // For c = a^b: dc/da = b * a^(b-1), dc/db = a^b * ln(a)
    assert_eq!(grads_pow.len(), 2);

    // Expected gradients for a=[2,3], b=[3,2]:
    // dc/da = [3*2^(3-1), 2*3^(2-1)] = [3*4, 2*3] = [12, 6]
    let expected_grad_a = &[12.0, 6.0];
    let actual_grad_a = grads_pow[0].as_ref().unwrap().as_slice().unwrap();

    // Check gradients with tolerance for floating point precision
    for (expected, actual) in expected_grad_a.iter().zip(actual_grad_a.iter()) {
        assert!(
            (expected - actual).abs() < 1e-6,
            "Power gradient mismatch: expected {}, got {}",
            expected,
            actual
        );
    }

    println!("Power gradients: da={:?}", actual_grad_a);
    // Note: Power operation fully implemented with proper gradient computation
}

#[test]
fn test_autograd_activation_functions() {
    let tape = GradientTape::new();

    let input = Tensor::<f32>::from_vec(vec![0.0, 1.0, -1.0], &[3]).unwrap();
    let input_tracked = tape.watch(input);

    // Test sigmoid
    let sigmoid_out = input_tracked.sigmoid().unwrap();
    let sigmoid_grads = tape.gradient(&[sigmoid_out], &[input_tracked]).unwrap();

    // For sigmoid: d(sigmoid(x))/dx = sigmoid(x) * (1 - sigmoid(x))
    assert_eq!(sigmoid_grads.len(), 1);
    let grad_data = sigmoid_grads[0].as_ref().unwrap().as_slice().unwrap();
    // sigmoid(0) = 0.5, gradient = 0.5 * 0.5 = 0.25
    assert!((grad_data[0] - 0.25).abs() < 1e-6);

    // For other values, just check they are positive (valid sigmoid gradients)
    assert!(grad_data[1] > 0.0);
    assert!(grad_data[2] > 0.0);
}

#[test]
fn test_autograd_tanh_function() {
    let tape = GradientTape::new();

    let input = Tensor::<f32>::from_vec(vec![0.0, 1.0], &[2]).unwrap();
    let input_tracked = tape.watch(input);

    // Test tanh
    let tanh_out = input_tracked.tanh().unwrap();
    let tanh_grads = tape.gradient(&[tanh_out], &[input_tracked]).unwrap();

    // For tanh: d(tanh(x))/dx = 1 - tanh^2(x)
    assert_eq!(tanh_grads.len(), 1);
    let grad_data = tanh_grads[0].as_ref().unwrap().as_slice().unwrap();
    // tanh(0) = 0, gradient = 1 - 0^2 = 1
    assert!((grad_data[0] - 1.0).abs() < 1e-6);

    // tanh(1) ≈ 0.76, gradient = 1 - 0.76^2 ≈ 0.42
    assert!(grad_data[1] > 0.0 && grad_data[1] < 1.0);
}

#[test]
fn test_autograd_softmax_function() {
    let tape = GradientTape::new();

    let input = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let input_tracked = tape.watch(input);

    // Test softmax
    let softmax_out = input_tracked.softmax(Some(-1)).unwrap();
    let softmax_grads = tape.gradient(&[softmax_out], &[input_tracked]).unwrap();

    // Softmax gradients should sum to approximately 0 (due to sum-to-1 constraint)
    assert_eq!(softmax_grads.len(), 1);
    let grad_data = softmax_grads[0].as_ref().unwrap().as_slice().unwrap();
    // For softmax, the sum of gradients should be approximately 0
    let grad_sum: f32 = grad_data.iter().sum();
    assert!(
        grad_sum.abs() < 1e-5,
        "Softmax gradients should sum to ~0, got {}",
        grad_sum
    );
}

#[test]
fn test_autograd_complex_computation() {
    let tape = GradientTape::new();

    let x = Tensor::<f32>::from_vec(vec![2.0], &[1]).unwrap();
    let y = Tensor::<f32>::from_vec(vec![3.0], &[1]).unwrap();

    let x_tracked = tape.watch(x);
    let y_tracked = tape.watch(y);

    // Complex computation: z = (x * y) / (x + y) + sigmoid(x - y)
    let xy = x_tracked.mul(&y_tracked).unwrap();
    let x_plus_y = x_tracked.add(&y_tracked).unwrap();
    let xy_div = xy.div(&x_plus_y).unwrap();

    let x_minus_y = x_tracked.sub(&y_tracked).unwrap();
    let sigmoid_part = x_minus_y.sigmoid().unwrap();

    let z = xy_div.add(&sigmoid_part).unwrap();

    let grads = tape.gradient(&[z], &[x_tracked, y_tracked]).unwrap();

    // Just verify that gradients exist and have reasonable values
    assert_eq!(grads.len(), 2);

    let grad_val_x = grads[0].as_ref().unwrap().as_slice().unwrap()[0];
    assert!(grad_val_x.is_finite());
    assert!(!grad_val_x.is_nan());

    let grad_val_y = grads[1].as_ref().unwrap().as_slice().unwrap()[0];
    assert!(grad_val_y.is_finite());
    assert!(!grad_val_y.is_nan());
}

#[test]
fn test_autograd_broadcasting_gradients() {
    let tape = GradientTape::new();

    let a = Tensor::<f32>::from_vec(vec![1.0, 2.0], &[2, 1]).unwrap();
    let b = Tensor::<f32>::from_vec(vec![3.0, 4.0], &[1, 2]).unwrap();

    let a_tracked = tape.watch(a);
    let b_tracked = tape.watch(b);

    // Test multiplication with broadcasting
    let c = a_tracked.mul(&b_tracked).unwrap();
    assert_eq!(c.tensor.shape().dims(), &[2, 2]); // Result shape after broadcasting

    let grads = tape.gradient(&[c], &[a_tracked, b_tracked]).unwrap();

    // Gradients should be unbroadcast to original shapes
    assert_eq!(grads.len(), 2);
    let grad_a = grads[0].as_ref().unwrap();
    let grad_b = grads[1].as_ref().unwrap();
    assert_eq!(grad_a.shape().dims(), &[2, 1]); // Original shape of a
    assert_eq!(grad_b.shape().dims(), &[1, 2]); // Original shape of b
}

#[test]
fn test_autograd_chained_operations() {
    let tape = GradientTape::new();

    let x = Tensor::<f32>::from_vec(vec![1.0, 2.0], &[2]).unwrap();
    let x_tracked = tape.watch(x);

    // Chain multiple operations: y = relu(x^2 - 1)
    let two = Tensor::<f32>::from_vec(vec![2.0, 2.0], &[2]).unwrap();
    let two_tracked = tape.watch(two);
    let x_squared = x_tracked.pow(&two_tracked).unwrap();
    let ones = Tensor::<f32>::from_vec(vec![1.0, 1.0], &[2]).unwrap();
    let ones_tracked = tape.watch(ones);

    let x_sq_minus_1 = x_squared.sub(&ones_tracked).unwrap();
    let y = x_sq_minus_1.relu().unwrap();

    let grads = tape.gradient(&[y], &[x_tracked]).unwrap();

    // Verify gradient exists and has correct shape
    assert_eq!(grads.len(), 1);
    let grad_tensor = grads[0].as_ref().unwrap();
    assert_eq!(grad_tensor.shape().dims(), &[2]);
    let grad_data = grad_tensor.as_slice().unwrap();

    // For x = [1, 2]:
    // x^2 = [1, 4]
    // x^2 - 1 = [0, 3]
    // relu([0, 3]) = [0, 3]
    // gradient of relu: [0, 1] (mask)
    // gradient of (x^2 - 1): [2*x, 2*x] = [2, 4]
    // combined: [0*2, 1*4] = [0, 4]
    assert_eq!(grad_data[0], 0.0);
    assert_eq!(grad_data[1], 4.0);
}
