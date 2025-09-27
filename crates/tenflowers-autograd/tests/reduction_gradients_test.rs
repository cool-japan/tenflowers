use tenflowers_autograd::GradientTape;
use tenflowers_core::Tensor;

#[test]
fn test_sum_gradient() {
    let tape = GradientTape::new();

    // Create input tensor [1.0, 2.0, 3.0, 4.0] -> shape [2, 2]
    let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let x = tape.watch(input);

    // Compute sum: should be 10.0
    let y = x.sum(None, false).unwrap();

    // Compute gradients
    let grads = tape.gradient(&[y], &[x]).unwrap();

    // For sum, all gradients should be 1.0
    assert_eq!(grads.len(), 1);
    let grad_tensor = grads[0].as_ref().unwrap();
    assert_eq!(grad_tensor.as_slice().unwrap(), &[1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn test_mean_gradient() {
    let tape = GradientTape::new();

    // Create input tensor [1.0, 2.0, 3.0, 4.0] -> shape [2, 2]
    let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let x = tape.watch(input);

    // Compute mean: should be 2.5
    let y = x.mean(None, false).unwrap();

    // Compute gradients
    let grads = tape.gradient(&[y], &[x]).unwrap();

    // For mean, all gradients should be 1/4 = 0.25
    assert_eq!(grads.len(), 1);
    let grad_tensor = grads[0].as_ref().unwrap();
    assert_eq!(grad_tensor.as_slice().unwrap(), &[0.25, 0.25, 0.25, 0.25]);
}

#[test]
fn test_sum_mean_chain() {
    let tape = GradientTape::new();

    // Create input tensor [1.0, 2.0, 3.0, 4.0]
    let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[4]).unwrap();
    let x = tape.watch(input);

    // Chain: sum then multiply by 2 then mean
    let sum_result = x.sum(None, false).unwrap();
    let two = tape.watch(Tensor::from_scalar(2.0f32));
    let doubled = sum_result.mul(&two).unwrap();

    // Compute gradients of doubled w.r.t. original input
    let grads = tape.gradient(&[doubled], &[x]).unwrap();

    // Gradient should be 2.0 for each element (since sum distributes 1 to each, then multiply by 2)
    assert_eq!(grads.len(), 1);
    let grad_tensor = grads[0].as_ref().unwrap();
    assert_eq!(grad_tensor.as_slice().unwrap(), &[2.0, 2.0, 2.0, 2.0]);
}
